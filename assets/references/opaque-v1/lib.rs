// Opaque sharded arena — first test version (legacy reference).
// Kept as the starting point for plugmem-arena; the v2 design lives in specs/01-arena.md.
// Dependency-free: core + alloc only.

//! Sharded memory-mapped arena for fixed-size types.
//!
//! This crate provides [`OpaqueBuckets`], a collection that stores elements in a flat byte array
//! partitioned into logical shards (buckets).
//!
//! # Mechanism
//! Elements are mapped to shards based on their sharding strategy. Each shard corresponds
//! to a fixed-size page in the underlying buffer. Sorting within shards is maintained
//! via memory shifts to enable binary search.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;

mod impls;

/// The number of elements a single bucket can store.
const SLOTS_PER_BUCKET: usize = 64;

/// Interface for types stored in [`OpaqueBuckets`].
pub trait OpaqueStorable: Clone {
    /// The byte size of the type.
    const SLOT_SIZE: usize;

    /// Returns the shard index for the instance.
    fn shard_index(&self, buckets: usize) -> usize;

    /// Returns the byte representation of the instance.
    fn as_bytes(&self) -> &[u8];

    /// Reconstructs the instance from a byte slice.
    fn from_bytes(bytes: &[u8]) -> Self;
}

/// A sharded collection that stores elements in a contiguous byte pool.
///
/// Elements are partitioned into pages based on shards.
///
/// # Generics
/// - `T`: The type stored in the arena.
/// - `B`: Number of shards.
/// - `M`: Maximum total capacity.
#[derive(Clone, PartialEq, Eq)]
pub struct OpaqueBuckets<T, const B: usize, const M: usize> {
    /// Linear buffer containing allocated pages.
    pool: Vec<u8>,
    /// Table mapping shard indices to physical page offsets.
    page_map: [u16; B],
    /// Array tracking the number of elements in each shard.
    counts: [u16; B],
    /// Total number of elements.
    total: u32,
    _marker: PhantomData<T>,
}

/// Iterator over elements in [`OpaqueBuckets`].
pub struct Iter<'a, T, const B: usize, const M: usize> {
    arena: &'a OpaqueBuckets<T, B, M>,
    current_shard: usize,
    current_index: usize,
    remaining: usize,
}

impl<T: OpaqueStorable, const B: usize, const M: usize> Default for OpaqueBuckets<T, B, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: OpaqueStorable, const B: usize, const M: usize> OpaqueBuckets<T, B, M> {
    /// Initializes an empty arena.
    pub fn new() -> Self {
        Self {
            pool: Vec::new(),
            page_map: [u16::MAX; B],
            counts: [0; B],
            total: 0,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    const fn page_size() -> usize {
        SLOTS_PER_BUCKET.saturating_mul(T::SLOT_SIZE)
    }

    /// Returns `true` if the element exists in the arena.
    pub fn contains(&self, id: &T) -> bool {
        let shard_idx = id.shard_index(B);
        let (page, count, _) = match self.get_page_context(shard_idx) {
            Some(ctx) => ctx,
            None => return false,
        };
        if count == 0 {
            return false;
        }
        Self::binary_search(page, count, id.as_bytes()).is_ok()
    }

    /// Inserts an element while maintaining order.
    pub fn insert(&mut self, id: T) -> Result<bool, error::Error> {
        if self.total >= M as u32 {
            return Err(error::Error::GlobalCapacityExceeded);
        }

        let shard_idx = id.shard_index(B);
        let page_idx = self.ensure_page(shard_idx)?;

        let (page, count, _) = self
            .get_page_context_mut(shard_idx)
            .expect("Page exists after ensure_page");

        match Self::binary_search(page, count, id.as_bytes()) {
            Ok(_) => Ok(false),
            Err(pos) => {
                if count >= SLOTS_PER_BUCKET {
                    return Err(error::Error::BucketCapacityExceeded);
                }

                self.insert_at(page_idx, pos, count, id.as_bytes());

                self.counts[shard_idx] = self.counts[shard_idx].saturating_add(1);
                self.total = self.total.saturating_add(1);
                Ok(true)
            }
        }
    }

    /// Removes an element.
    pub fn remove(&mut self, id: &T) -> bool {
        let shard_idx = id.shard_index(B);
        let (page, count, _) = match self.get_page_context_mut(shard_idx) {
            Some(ctx) => ctx,
            None => return false,
        };
        if count == 0 {
            return false;
        }

        if let Ok(pos) = Self::binary_search(page, count, id.as_bytes()) {
            let page_idx = self.page_map[shard_idx];
            self.remove_at(page_idx, pos, count);

            self.counts[shard_idx] = self.counts[shard_idx].saturating_sub(1);
            self.total = self.total.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Returns a mutable reference to the element's bytes.
    pub fn find_slice_mut_by<F>(&mut self, shard_idx: usize, f: F) -> Option<&mut [u8]>
    where
        F: FnMut(&[u8]) -> core::cmp::Ordering,
    {
        let (page, count, _) = self.get_page_context_mut(shard_idx)?;
        if count == 0 {
            return None;
        }
        Self::binary_search_by(page, count, f).ok().map(move |pos| {
            let offset = pos.saturating_mul(T::SLOT_SIZE);
            &mut page[offset..offset.saturating_add(T::SLOT_SIZE)]
        })
    }

    /// Returns a reference to the element's bytes.
    pub fn find_slice_by<F>(&self, shard_idx: usize, f: F) -> Option<&[u8]>
    where
        F: FnMut(&[u8]) -> core::cmp::Ordering,
    {
        let (page, count, _) = self.get_page_context(shard_idx)?;
        if count == 0 {
            return None;
        }
        Self::binary_search_by(page, count, f).ok().map(|pos| {
            let offset = pos.saturating_mul(T::SLOT_SIZE);
            &page[offset..offset.saturating_add(T::SLOT_SIZE)]
        })
    }

    /// Returns the element if it matches the criteria.
    pub fn find_by<F>(&self, shard_idx: usize, f: F) -> Option<T>
    where
        F: FnMut(&[u8]) -> core::cmp::Ordering,
    {
        let (page, count, _) = self.get_page_context(shard_idx)?;
        if count == 0 {
            return None;
        }
        Self::binary_search_by(page, count, f).ok().map(|pos| {
            let offset = pos.saturating_mul(T::SLOT_SIZE);
            T::from_bytes(&page[offset..offset.saturating_add(T::SLOT_SIZE)])
        })
    }

    /// Removes an element matching the criteria.
    pub fn remove_by<F>(&mut self, shard_idx: usize, f: F) -> Option<T>
    where
        F: FnMut(&[u8]) -> core::cmp::Ordering,
    {
        let (page, count, _) = self.get_page_context_mut(shard_idx)?;
        if count == 0 {
            return None;
        }
        let pos = Self::binary_search_by(page, count, f).ok()?;

        let offset = pos.saturating_mul(T::SLOT_SIZE);
        let item = T::from_bytes(&page[offset..offset.saturating_add(T::SLOT_SIZE)]);

        let page_idx = self.page_map[shard_idx];
        self.remove_at(page_idx, pos, count);
        self.counts[shard_idx] = self.counts[shard_idx].saturating_sub(1);
        self.total = self.total.saturating_sub(1);

        Some(item)
    }

    /// Returns the element if it exists.
    pub fn get(&self, target: &T) -> Option<T> {
        let shard_idx = target.shard_index(B);
        let (page, count, _) = self.get_page_context(shard_idx)?;
        if count == 0 {
            return None;
        }

        Self::binary_search(page, count, target.as_bytes())
            .ok()
            .map(|_| target.clone())
    }

    fn binary_search_by<F>(page: &[u8], len: usize, mut f: F) -> Result<usize, usize>
    where
        F: FnMut(&[u8]) -> core::cmp::Ordering,
    {
        let mut left = 0;
        let mut right = len;
        while left < right {
            let mid = left.saturating_add(right.saturating_sub(left) / 2);
            let offset = mid.saturating_mul(T::SLOT_SIZE);
            let probe = &page[offset..offset.saturating_add(T::SLOT_SIZE)];
            match f(probe) {
                core::cmp::Ordering::Less => left = mid.saturating_add(1),
                core::cmp::Ordering::Greater => right = mid,
                core::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Err(left)
    }

    #[inline(always)]
    fn binary_search(page: &[u8], len: usize, target: &[u8]) -> Result<usize, usize> {
        Self::binary_search_by(page, len, |probe| probe.cmp(target))
    }

    #[inline(always)]
    fn get_page_context_mut(&mut self, shard_idx: usize) -> Option<(&mut [u8], usize, usize)> {
        if shard_idx >= B {
            return None;
        }
        let page_idx = self.page_map[shard_idx];
        if page_idx == u16::MAX {
            return None;
        }

        let count = self.counts[shard_idx] as usize;
        let page_size = Self::page_size();
        let page_start = (page_idx as usize).saturating_mul(page_size);
        let page = unsafe {
            self.pool
                .get_unchecked_mut(page_start..page_start + page_size)
        };

        Some((page, count, page_start))
    }

    #[inline(always)]
    fn get_page_context(&self, shard_idx: usize) -> Option<(&[u8], usize, usize)> {
        if shard_idx >= B {
            return None;
        }
        let page_idx = self.page_map[shard_idx];
        if page_idx == u16::MAX {
            return None;
        }

        let count = self.counts[shard_idx] as usize;
        let page_size = Self::page_size();
        let page_start = (page_idx as usize).saturating_mul(page_size);
        let page = unsafe { self.pool.get_unchecked(page_start..page_start + page_size) };

        Some((page, count, page_start))
    }

    fn ensure_page(&mut self, shard_idx: usize) -> Result<u16, error::Error> {
        let mut page_idx = self.page_map[shard_idx];

        if page_idx == u16::MAX {
            let current_pool_size = self.pool.len();
            let new_page_idx = (current_pool_size / Self::page_size()) as u16;

            if (new_page_idx as usize)
                .saturating_add(1)
                .saturating_mul(Self::page_size())
                > (u16::MAX as usize) * Self::page_size()
            {
                return Err(error::Error::ArenaOverflow);
            }

            self.pool.reserve(Self::page_size());
            #[allow(clippy::uninit_vec)]
            unsafe {
                self.pool
                    .set_len(current_pool_size.saturating_add(Self::page_size()));
            }

            self.page_map[shard_idx] = new_page_idx;
            page_idx = new_page_idx;
        }

        Ok(page_idx)
    }

    fn insert_at(&mut self, page_idx: u16, pos: usize, current_count: usize, target: &[u8]) {
        let page_start = (page_idx as usize).saturating_mul(Self::page_size());
        let byte_pos = page_start.saturating_add(pos.saturating_mul(T::SLOT_SIZE));
        let bytes_to_shift = current_count
            .saturating_sub(pos)
            .saturating_mul(T::SLOT_SIZE);

        if bytes_to_shift > 0 {
            self.pool.copy_within(
                byte_pos..byte_pos.saturating_add(bytes_to_shift),
                byte_pos.saturating_add(T::SLOT_SIZE),
            );
        }

        self.pool[byte_pos..byte_pos.saturating_add(T::SLOT_SIZE)].copy_from_slice(target);
    }

    fn remove_at(&mut self, page_idx: u16, pos: usize, current_count: usize) {
        let page_start = (page_idx as usize).saturating_mul(Self::page_size());
        let byte_pos = page_start.saturating_add(pos.saturating_mul(T::SLOT_SIZE));
        let bytes_to_shift = current_count
            .saturating_sub(pos)
            .saturating_sub(1)
            .saturating_mul(T::SLOT_SIZE);

        if bytes_to_shift > 0 {
            let src_start = byte_pos.saturating_add(T::SLOT_SIZE);
            self.pool.copy_within(
                src_start..src_start.saturating_add(bytes_to_shift),
                byte_pos,
            );
        }
    }

    /// Returns the total element count.
    pub fn len(&self) -> u32 {
        self.total
    }

    /// Returns `true` if empty.
    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Returns an iterator over all elements.
    pub fn iter(&self) -> Iter<'_, T, B, M> {
        Iter {
            arena: self,
            current_shard: 0,
            current_index: 0,
            remaining: self.total as usize,
        }
    }
}

impl<'a, T: OpaqueStorable, const B: usize, const M: usize> IntoIterator
    for &'a OpaqueBuckets<T, B, M>
{
    type Item = T;
    type IntoIter = Iter<'a, T, B, M>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: OpaqueStorable, const B: usize, const M: usize> Iterator for Iter<'a, T, B, M> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.current_shard < B {
            let count = self.arena.counts[self.current_shard] as usize;

            if self.current_index < count {
                let page_idx = self.arena.page_map[self.current_shard];
                let page_start =
                    (page_idx as usize).saturating_mul(OpaqueBuckets::<T, B, M>::page_size());
                let offset =
                    page_start.saturating_add(self.current_index.saturating_mul(T::SLOT_SIZE));

                let item =
                    T::from_bytes(&self.arena.pool[offset..offset.saturating_add(T::SLOT_SIZE)]);
                self.current_index = self.current_index.saturating_add(1);
                self.remaining = self.remaining.saturating_sub(1);
                return Some(item);
            } else {
                self.current_shard = self.current_shard.saturating_add(1);
                self.current_index = 0;
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a, T: OpaqueStorable, const B: usize, const M: usize> ExactSizeIterator
    for Iter<'a, T, B, M>
{
}

impl<T: OpaqueStorable, const B: usize, const M: usize> fmt::Debug for OpaqueBuckets<T, B, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpaqueBuckets")
            .field("total", &self.total)
            .field("buckets", &B)
            .field("allocated_pages", &(self.pool.len() / Self::page_size()))
            .finish()
    }
}

/// Errors occurring during [`OpaqueBuckets`] operations.
pub mod error {
    use core::fmt;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Error {
        /// Bucket capacity reached (max 64 slots).
        BucketCapacityExceeded,
        /// Global capacity reached.
        GlobalCapacityExceeded,
        /// Internal index limit reached.
        ArenaOverflow,
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Error::BucketCapacityExceeded => f.write_str("Bucket capacity exceeded"),
                Error::GlobalCapacityExceeded => f.write_str("Global capacity exceeded"),
                Error::ArenaOverflow => f.write_str("Arena overflow"),
            }
        }
    }

    impl core::error::Error for Error {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn create_id(id_byte: u8, sub_byte: u8) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[0] = id_byte;
        bytes[31] = sub_byte;
        bytes
    }

    #[test]
    fn test_basic_lifecycle() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let id1 = create_id(10, 1);
        let id2 = create_id(10, 2);
        let id3 = create_id(20, 1);

        assert!(arena.insert(id1).unwrap());
        assert!(arena.insert(id2).unwrap());
        assert!(arena.insert(id3).unwrap());
        assert!(arena.contains(&id1));
        assert!(arena.contains(&id2));
        assert!(arena.contains(&id3));
        assert_eq!(arena.len(), 3);

        assert!(!arena.insert(id1).unwrap());
        assert_eq!(arena.len(), 3);

        assert!(arena.remove(&id2));
        assert!(!arena.contains(&id2));
        assert_eq!(arena.len(), 2);
    }

    #[test]
    fn test_bucket_overflow() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 1000>::new();
        for i in 0..64 {
            let id = create_id(5, i as u8);
            assert!(arena.insert(id).unwrap());
        }

        let overflow_id = create_id(5, 65);
        let result = arena.insert(overflow_id);
        assert!(matches!(result, Err(error::Error::BucketCapacityExceeded)));
    }

    #[test]
    fn test_global_capacity() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 10>::new();
        for i in 0..10 {
            let id = create_id(i as u8, 0);
            assert!(arena.insert(id).unwrap());
        }

        let overflow_id = create_id(255, 255);
        let result = arena.insert(overflow_id);
        assert!(matches!(result, Err(error::Error::GlobalCapacityExceeded)));
    }

    #[test]
    fn test_lazy_allocation() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        assert_eq!(arena.pool.len(), 0);

        arena.insert(create_id(1, 0)).unwrap();
        assert_eq!(arena.pool.len(), 2048);

        arena.insert(create_id(2, 0)).unwrap();
        assert_eq!(arena.pool.len(), 4096);
    }

    #[test]
    fn test_sorted_binary_search() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        for i in (0..10).rev() {
            let id = create_id(10, i as u8);
            arena.insert(id).unwrap();
        }

        for i in 0..10 {
            let id = create_id(10, i as u8);
            assert!(arena.contains(&id));
        }
    }

    #[test]
    fn test_iteration() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let ids = [create_id(1, 1), create_id(50, 1), create_id(255, 1)];

        for id in &ids {
            arena.insert(*id).unwrap();
        }

        let result: Vec<_> = arena.iter().collect();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], ids[0]);
        assert_eq!(result[1], ids[1]);
        assert_eq!(result[2], ids[2]);
    }

    #[test]
    fn test_removal_with_shifting() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let shard = 10;
        let id1 = create_id(shard, 1);
        let id2 = create_id(shard, 2);
        let id3 = create_id(shard, 3);

        arena.insert(id1).unwrap();
        arena.insert(id2).unwrap();
        arena.insert(id3).unwrap();

        assert!(arena.remove(&id2));
        assert!(!arena.contains(&id2));
        assert!(arena.contains(&id1));
        assert!(arena.contains(&id3));
        assert_eq!(arena.len(), 2);

        let remaining: Vec<_> = arena.iter().collect();
        assert_eq!(remaining, vec![id1, id3]);
    }

    #[test]
    fn test_removal_boundaries() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let shard = 42;
        let id_first = create_id(shard, 1);
        let id_mid = create_id(shard, 2);
        let id_last = create_id(shard, 3);

        arena.insert(id_first).unwrap();
        arena.insert(id_mid).unwrap();
        arena.insert(id_last).unwrap();

        assert!(arena.remove(&id_first));
        assert!(!arena.contains(&id_first));
        assert!(arena.contains(&id_mid));
        assert!(arena.contains(&id_last));

        assert!(arena.remove(&id_last));
        assert!(!arena.contains(&id_last));
        assert!(arena.contains(&id_mid));

        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn test_reinsertion_cycle() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let id = create_id(7, 7);

        assert!(arena.insert(id).unwrap());
        assert!(arena.remove(&id));
        assert!(!arena.contains(&id));

        assert!(arena.insert(id).unwrap());
        assert!(arena.contains(&id));
        assert_eq!(arena.len(), 1);
    }

    #[test]
    fn test_clone_and_partial_eq() {
        let mut arena1 = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        arena1.insert(create_id(1, 1)).unwrap();
        arena1.insert(create_id(2, 2)).unwrap();

        let arena2 = arena1.clone();
        assert_eq!(arena1, arena2);
        assert!(arena2.contains(&create_id(1, 1)));
        assert!(arena2.contains(&create_id(2, 2)));
    }

    #[test]
    fn test_all_shards_allocation() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 256>::new();
        for i in 0..256 {
            let id = create_id(i as u8, 0);
            assert!(arena.insert(id).unwrap());
        }

        assert_eq!(arena.len(), 256);
        assert_eq!(arena.pool.len(), 256 * 2048);

        for i in 0..256 {
            let id = create_id(i as u8, 0);
            assert!(arena.contains(&id));
        }
    }

    #[test]
    fn test_duplicate_removal() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let id = create_id(1, 1);

        arena.insert(id).unwrap();
        assert!(arena.remove(&id));
        assert!(!arena.remove(&id));
        assert_eq!(arena.len(), 0);
    }

    #[test]
    fn test_empty_iteration() {
        let arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        assert_eq!(arena.iter().count(), 0);
    }

    #[test]
    fn test_contains_unallocated_shard() {
        let arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let id = create_id(5, 5);
        assert!(!arena.contains(&id));
    }

    #[test]
    fn test_find_by_partial_search() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let shard = 5;
        let id = create_id(shard, 42);
        arena.insert(id).unwrap();

        let found = arena.find_by(shard as usize, |probe| probe[31].cmp(&42));
        assert_eq!(found, Some(id));
    }

    #[test]
    fn test_remove_by_partial_search() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let shard = 5;
        let id = create_id(shard, 42);
        arena.insert(id).unwrap();

        let removed = arena.remove_by(shard as usize, |probe| probe[31].cmp(&42));
        assert_eq!(removed, Some(id));
        assert!(!arena.contains(&id));
    }

    #[test]
    fn test_insert_middle_shifting() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let shard = 15;
        let id_large = create_id(shard, 30);
        let id_small = create_id(shard, 10);
        let id_mid = create_id(shard, 20);

        arena.insert(id_large).unwrap();
        arena.insert(id_small).unwrap();
        arena.insert(id_mid).unwrap();

        let items: Vec<_> = arena.iter().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], id_small);
        assert_eq!(items[1], id_mid);
        assert_eq!(items[2], id_large);
    }

    #[test]
    fn test_api_get_and_slices() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let shard = 5;
        let id = create_id(shard, 42);
        arena.insert(id).unwrap();

        assert_eq!(arena.get(&id), Some(id));
        assert_eq!(arena.get(&create_id(shard, 99)), None);

        let slice = arena.find_slice_by(shard as usize, |probe| probe[31].cmp(&42));
        assert!(slice.is_some());
        assert_eq!(slice.unwrap().len(), 32);
        assert_eq!(slice.unwrap()[31], 42);

        if let Some(mut_slice) = arena.find_slice_mut_by(shard as usize, |probe| probe[31].cmp(&42))
        {
            mut_slice[30] = 99;
        }

        let updated_id = arena
            .find_by(shard as usize, |probe| probe[31].cmp(&42))
            .unwrap();
        assert_eq!(updated_id[30], 99);
    }

    #[test]
    fn test_is_empty_and_exact_size_iterator() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        assert!(arena.is_empty());
        assert_eq!(arena.iter().len(), 0);

        let id = create_id(1, 1);
        arena.insert(id).unwrap();
        assert!(!arena.is_empty());
        assert_eq!(arena.iter().len(), 1);

        arena.remove(&id);
        assert!(arena.is_empty());
        assert_eq!(arena.iter().len(), 0);
    }

    #[test]
    fn test_cross_shard_iteration_with_gaps() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 100>::new();
        let id_shard_2 = create_id(2, 1);
        let id_shard_250 = create_id(250, 1);

        arena.insert(id_shard_250).unwrap();
        arena.insert(id_shard_2).unwrap();

        let mut iter = arena.iter();
        assert_eq!(iter.next(), Some(id_shard_2));
        assert_eq!(iter.next(), Some(id_shard_250));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_zero_global_capacity() {
        let mut arena = OpaqueBuckets::<[u8; 32], 256, 0>::new();
        let id = create_id(1, 1);
        let result = arena.insert(id);
        assert!(matches!(result, Err(error::Error::GlobalCapacityExceeded)));
        assert_eq!(arena.pool.len(), 0);
    }
}
