// Slot implementations for the first test version of the opaque arena (legacy reference).
// Only plain core types: fixed-size byte arrays, integers, NonZero wrappers.

use crate::OpaqueStorable;

// 1. Fixed-size byte identifiers (20- and 32-byte ids / hashes)
// -------------------------------------------------------------------------

macro_rules! impl_storable_for_bytes {
    ($($n:literal),*) => {
        $(
            impl OpaqueStorable for [u8; $n] {
                const SLOT_SIZE: usize = $n;
                #[inline(always)]
                fn shard_index(&self, buckets: usize) -> usize {
                    // Use byte 0 for sharding. This ensures range-based sorting order.
                    (self[0] as usize).saturating_mul(buckets) / 256
                }
                #[inline(always)]
                fn as_bytes(&self) -> &[u8] { self }
                #[inline(always)]
                fn from_bytes(bytes: &[u8]) -> Self {
                    let mut arr = [0u8; $n];
                    arr.copy_from_slice(bytes);
                    arr
                }
            }
        )*
    };
}

impl_storable_for_bytes!(20, 32);

// 2. Integers
// -------------------------------------------------------------------------

macro_rules! impl_storable_for_num {
    ($($t:ty),*) => {
        $(
            impl OpaqueStorable for $t {
                const SLOT_SIZE: usize = core::mem::size_of::<$t>();
                #[inline(always)]
                fn shard_index(&self, buckets: usize) -> usize {
                    (*self as u128 as usize) % buckets
                }
                #[inline(always)]
                fn as_bytes(&self) -> &[u8] {
                    unsafe { core::slice::from_raw_parts(self as *const $t as *const u8, Self::SLOT_SIZE) }
                }
                #[inline(always)]
                fn from_bytes(bytes: &[u8]) -> Self {
                    let mut arr = [0u8; core::mem::size_of::<$t>()];
                    arr.copy_from_slice(bytes);
                    <$t>::from_le_bytes(arr)
                }
            }
        )*
    };
}

impl_storable_for_num!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128);

// 3. NonZero Wrappers
// -------------------------------------------------------------------------

macro_rules! impl_storable_for_nonzero {
    ($($t:ty, $base:ty),*) => {
        $(
            impl OpaqueStorable for $t {
                const SLOT_SIZE: usize = core::mem::size_of::<$t>();
                #[inline(always)]
                fn shard_index(&self, buckets: usize) -> usize {
                    (self.get() as u128 as usize) % buckets
                }
                #[inline(always)]
                fn as_bytes(&self) -> &[u8] {
                    unsafe { core::slice::from_raw_parts(self as *const $t as *const u8, Self::SLOT_SIZE) }
                }
                #[inline(always)]
                fn from_bytes(bytes: &[u8]) -> Self {
                    let mut arr = [0u8; core::mem::size_of::<$t>()];
                    arr.copy_from_slice(bytes);
                    Self::new(<$base>::from_le_bytes(arr)).expect("NonZero value is zero")
                }
            }
        )*
    };
}

impl_storable_for_nonzero!(
    core::num::NonZeroU8,
    u8,
    core::num::NonZeroU16,
    u16,
    core::num::NonZeroU32,
    u32,
    core::num::NonZeroU64,
    u64,
    core::num::NonZeroU128,
    u128
);

// 4. Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::{OpaqueBuckets, OpaqueStorable};

    #[test]
    fn test_slot_sizes() {
        assert_eq!(u8::SLOT_SIZE, 1);
        assert_eq!(u16::SLOT_SIZE, 2);
        assert_eq!(u32::SLOT_SIZE, 4);
        assert_eq!(u64::SLOT_SIZE, 8);
        assert_eq!(u128::SLOT_SIZE, 16);
        assert_eq!(i8::SLOT_SIZE, 1);
        assert_eq!(i16::SLOT_SIZE, 2);
        assert_eq!(i32::SLOT_SIZE, 4);
        assert_eq!(i64::SLOT_SIZE, 8);
        assert_eq!(i128::SLOT_SIZE, 16);
        assert_eq!(<[u8; 20]>::SLOT_SIZE, 20);
        assert_eq!(<[u8; 32]>::SLOT_SIZE, 32);
    }

    macro_rules! test_numeric_type {
        ($t:ty, $name:ident, $v1:expr, $v2:expr) => {
            #[test]
            fn $name() {
                let mut arena = OpaqueBuckets::<$t, 16, 100>::new();
                let values = [<$t>::MIN, <$t>::MAX, $v1, $v2];

                for &val in &values {
                    let restored = <$t>::from_bytes(val.as_bytes());
                    assert_eq!(val, restored, "SerDe error for {}", stringify!($t));
                    if !arena.contains(&val) {
                        assert!(arena.insert(val).unwrap());
                    }
                }

                for &val in &values {
                    assert!(arena.contains(&val));
                    arena.remove(&val);
                    assert!(!arena.contains(&val));
                }
            }
        };
    }

    test_numeric_type!(u8, test_u8, 1, 42);
    test_numeric_type!(u16, test_u16, 1000, 4242);
    test_numeric_type!(u32, test_u32, 100000, 1234567);
    test_numeric_type!(u64, test_u64, 1000000000, 0x1122334455667788);
    test_numeric_type!(
        u128,
        test_u128,
        1000000000000,
        0x11223344556677889900AABBCCDDEEFF
    );

    test_numeric_type!(i8, test_i8, -1, 42);
    test_numeric_type!(i16, test_i16, -1000, 1000);
    test_numeric_type!(i32, test_i32, -100000, 100000);
    test_numeric_type!(i64, test_i64, -1000000000, 1000000000);
    test_numeric_type!(i128, test_i128, -1000000000000, 1000000000000);

    #[test]
    fn test_nonzero_types() {
        use core::num::NonZeroU32;
        let mut arena = OpaqueBuckets::<NonZeroU32, 16, 10>::new();
        let val = NonZeroU32::new(42).unwrap();
        assert!(arena.insert(val).unwrap());
        assert!(arena.contains(&val));
    }

    #[test]
    fn test_byte_id_complete() {
        let mut arena = OpaqueBuckets::<[u8; 32], 16, 10>::new();
        let mut id = [0u8; 32];
        id[0] = 0xAB;
        id[31] = 0xCD;
        assert!(arena.insert(id).unwrap());
        assert!(arena.contains(&id));
        assert_eq!(<[u8; 32]>::from_bytes(id.as_bytes()), id);
    }

    #[test]
    fn test_short_byte_id() {
        let mut arena = OpaqueBuckets::<[u8; 20], 16, 10>::new();
        let mut id = [0u8; 20];
        id[0] = 0xEF;
        assert!(arena.insert(id).unwrap());
        assert!(arena.contains(&id));
    }
}
