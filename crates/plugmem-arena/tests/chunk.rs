//! Boundary tests for `ChunkPool` (specs/01 test plan) plus a property
//! model against `Vec<Vec<Vec<u8>>>` (lists of values).

use plugmem_arena::{CHUNK_BYTES, CHUNK_PAYLOAD, ChunkPool, ChunkPoolCfg, Error, ListHandle};
use proptest::prelude::*;

/// Concatenated bytes of one list, as the consumers (varint decoders) see
/// them.
fn bytes_of(pool: &ChunkPool, list: &ListHandle) -> Vec<u8> {
    pool.iter(list).flatten().copied().collect()
}

#[test]
fn empty_pool_and_empty_handle() {
    let pool = ChunkPool::new(ChunkPoolCfg::new());
    let list = ListHandle::EMPTY;
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
    assert_eq!(pool.iter(&list).count(), 0);
    assert_eq!(pool.pool_bytes(), 0);
    assert_eq!(ListHandle::default(), ListHandle::EMPTY);
}

#[test]
fn single_value_roundtrip() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, b"hello").unwrap();
    assert_eq!(list.len(), 1);
    assert!(!list.is_empty());
    assert_eq!(bytes_of(&pool, &list), b"hello");
    assert_eq!(pool.pool_bytes(), CHUNK_BYTES);
}

#[test]
fn values_filling_exactly_one_chunk() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    for i in 0..3u8 {
        pool.push(&mut list, &[i; CHUNK_PAYLOAD / 3]).unwrap();
    }
    // 3 x 20 = 60 bytes: exactly one full chunk, no second allocation.
    assert_eq!(pool.pool_bytes(), CHUNK_BYTES);
    let chunks: Vec<&[u8]> = pool.iter(&list).collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].len(), CHUNK_PAYLOAD);
}

#[test]
fn value_that_does_not_fit_starts_a_fresh_chunk() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, &[1; 50]).unwrap();
    pool.push(&mut list, &[2; 20]).unwrap(); // 50 + 20 > 60 -> new chunk
    let chunks: Vec<&[u8]> = pool.iter(&list).collect();
    // The value never straddles chunks; the 10 skipped tail bytes of the
    // first chunk are invisible to iteration.
    assert_eq!(chunks, vec![[1u8; 50].as_slice(), [2u8; 20].as_slice()]);
    assert_eq!(bytes_of(&pool, &list).len(), 70);
    assert_eq!(pool.pool_bytes(), 2 * CHUNK_BYTES);
}

#[test]
fn value_size_boundaries() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, &[9; CHUNK_PAYLOAD]).unwrap(); // exactly max fits
    assert_eq!(
        pool.push(&mut list, &[9; CHUNK_PAYLOAD + 1]),
        Err(Error::ValueTooLarge {
            len: CHUNK_PAYLOAD + 1
        })
    );
    // The failed push left both the pool and the handle unchanged.
    assert_eq!(list.len(), 1);
    assert_eq!(bytes_of(&pool, &list).len(), CHUNK_PAYLOAD);
}

#[test]
fn empty_value_bumps_len_without_bytes() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, b"").unwrap();
    assert_eq!(list.len(), 1);
    assert_eq!(pool.pool_bytes(), 0); // no chunk allocated
    assert_eq!(pool.iter(&list).count(), 0);
    pool.push(&mut list, b"x").unwrap();
    pool.push(&mut list, b"").unwrap();
    assert_eq!(list.len(), 3);
    assert_eq!(bytes_of(&pool, &list), b"x");
}

#[test]
fn interleaved_lists_are_independent() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut a = ListHandle::EMPTY;
    let mut b = ListHandle::EMPTY;
    for i in 0..100u32 {
        pool.push(&mut a, &i.to_be_bytes()).unwrap();
        pool.push(&mut b, &(i * 2).to_be_bytes()).unwrap();
    }
    let back = |bytes: Vec<u8>| -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
            .collect()
    };
    assert_eq!(back(bytes_of(&pool, &a)), (0..100).collect::<Vec<u32>>());
    assert_eq!(
        back(bytes_of(&pool, &b)),
        (0..100).map(|i| i * 2).collect::<Vec<u32>>()
    );
}

#[test]
fn free_recycles_chunks_and_resets_the_handle() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    for i in 0..200u32 {
        pool.push(&mut list, &i.to_be_bytes()).unwrap();
    }
    let peak = pool.pool_bytes();
    pool.free(&mut list);
    assert_eq!(list, ListHandle::EMPTY);
    // Refill (same volume, another handle): the pool must reuse the freed
    // chain instead of growing.
    let mut other = ListHandle::EMPTY;
    for i in 0..200u32 {
        pool.push(&mut other, &i.to_be_bytes()).unwrap();
    }
    assert_eq!(pool.pool_bytes(), peak);
    assert_eq!(bytes_of(&pool, &other).len(), 800);
}

#[test]
fn free_of_an_empty_list_is_a_no_op() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.free(&mut list);
    assert_eq!(list, ListHandle::EMPTY);
    assert_eq!(pool.pool_bytes(), 0);
}

#[test]
fn freeing_one_list_leaves_others_intact() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut doomed = ListHandle::EMPTY;
    let mut kept = ListHandle::EMPTY;
    for i in 0..50u32 {
        pool.push(&mut doomed, &i.to_be_bytes()).unwrap();
        pool.push(&mut kept, &i.to_be_bytes()).unwrap();
    }
    let want = bytes_of(&pool, &kept);
    pool.free(&mut doomed);
    // Overwrite the recycled chunks through a third list.
    let mut noise = ListHandle::EMPTY;
    for _ in 0..50u32 {
        pool.push(&mut noise, &[0xAA; 4]).unwrap();
    }
    assert_eq!(bytes_of(&pool, &kept), want);
}

#[test]
fn max_bytes_boundary() {
    // Room for exactly two chunks.
    let mut pool = ChunkPool::new(ChunkPoolCfg::new().with_max_bytes(2 * CHUNK_BYTES));
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, &[1; CHUNK_PAYLOAD]).unwrap();
    pool.push(&mut list, &[2; CHUNK_PAYLOAD]).unwrap();
    assert_eq!(
        pool.push(&mut list, &[3; 1]),
        Err(Error::CapacityExceeded {
            max_bytes: 2 * CHUNK_BYTES
        })
    );
    // Freeing returns chunks; pushing then succeeds without growth.
    pool.free(&mut list);
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, &[4; 1]).unwrap();
    assert_eq!(pool.pool_bytes(), 2 * CHUNK_BYTES);
}

#[test]
fn debug_is_a_summary() {
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, b"secret").unwrap();
    let dump = format!("{pool:?}");
    assert!(dump.contains("chunks: 1"));
    assert!(!dump.contains("secret"));
}

#[test]
fn cfg_default_and_builder() {
    assert_eq!(ChunkPoolCfg::default(), ChunkPoolCfg::new());
    assert_eq!(ChunkPoolCfg::new().with_max_bytes(64).max_bytes, 64);
}

#[test]
fn handle_byte_encoding_is_stable() {
    // The wire form is a format contract: [head BE | tail BE | len BE].
    // EMPTY has head = tail = NONE (u32::MAX) and len = 0.
    assert_eq!(
        ListHandle::EMPTY.to_bytes(),
        [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0, 0]
    );
    assert_eq!(
        ListHandle::from_bytes(ListHandle::EMPTY.to_bytes()),
        ListHandle::EMPTY
    );

    // A populated handle survives the roundtrip and keeps working with its
    // pool: chunk 0 is both head and tail after two small pushes.
    let mut pool = ChunkPool::new(ChunkPoolCfg::new());
    let mut list = ListHandle::EMPTY;
    pool.push(&mut list, b"ab").unwrap();
    pool.push(&mut list, b"cd").unwrap();
    let restored = ListHandle::from_bytes(list.to_bytes());
    assert_eq!(restored, list);
    assert_eq!(
        list.to_bytes(),
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
        "head = tail = chunk 0, len = 2"
    );
    assert_eq!(bytes_of(&pool, &restored), b"abcd");
}

/// One step of the property workload.
#[derive(Debug, Clone)]
enum Op {
    /// Start a new list.
    New,
    /// Push a value into list `i % live_lists`.
    Push(usize, Vec<u8>),
    /// Free list `i % live_lists` (its slot stays, emptied).
    Free(usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        1 => Just(Op::New),
        8 => (any::<usize>(), proptest::collection::vec(any::<u8>(), 0..=CHUNK_PAYLOAD))
            .prop_map(|(i, v)| Op::Push(i, v)),
        1 => any::<usize>().prop_map(Op::Free),
    ]
}

proptest! {
    /// The pool must behave exactly like independent `Vec<Vec<u8>>` lists:
    /// per list, the concatenated iterated bytes equal the concatenated
    /// pushed values, and no value ever straddles two yielded slices.
    #[test]
    // proptest's harness calls into the OS (cwd for failure
    // persistence), which miri's isolation forbids; UB-paths are covered
    // by the boundary tests.
    #[cfg_attr(miri, ignore)]
    fn behaves_like_vec_of_lists(ops in proptest::collection::vec(op_strategy(), 1..200)) {
        let mut pool = ChunkPool::new(ChunkPoolCfg::new());
        let mut handles: Vec<ListHandle> = vec![ListHandle::EMPTY];
        let mut model: Vec<Vec<Vec<u8>>> = vec![Vec::new()];

        for op in ops {
            match op {
                Op::New => {
                    handles.push(ListHandle::EMPTY);
                    model.push(Vec::new());
                }
                Op::Push(i, value) => {
                    let i = i % handles.len();
                    pool.push(&mut handles[i], &value).unwrap();
                    model[i].push(value);
                }
                Op::Free(i) => {
                    let i = i % handles.len();
                    pool.free(&mut handles[i]);
                    model[i].clear();
                }
            }
        }

        for (handle, values) in handles.iter().zip(&model) {
            prop_assert_eq!(handle.len() as usize, values.len());
            let want: Vec<u8> = values.iter().flatten().copied().collect();
            let got: Vec<u8> = pool.iter(handle).flatten().copied().collect();
            prop_assert_eq!(got, want);
            // Boundary invariant: walking the yielded slices, every value
            // must be fully contained in a single slice.
            let mut slices = pool.iter(handle);
            let mut remaining: &[u8] = &[];
            for value in values.iter().filter(|v| !v.is_empty()) {
                if remaining.len() < value.len() {
                    prop_assert!(remaining.is_empty(), "value split across chunks");
                    remaining = slices.next().unwrap();
                }
                prop_assert_eq!(&remaining[..value.len()], value.as_slice());
                remaining = &remaining[value.len()..];
            }
        }
    }
}
