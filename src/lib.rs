//! # kdtree-na
//!
//! K-dimensional tree for Rust (bucket point-region implementation)
//!
//! ## Usage
//!
//! ```
//! extern crate nalgebra;
//!
//! use nalgebra::{vector, Vector2};
//! use kdtree_na::{norm::EuclideanNormSquared, KdTree};
//!
//! let a: (Vector2<f64>, usize) = (vector![0f64, 0f64], 0);
//! let b: (Vector2<f64>, usize) = (vector![1f64, 1f64], 1);
//! let c: (Vector2<f64>, usize) = (vector![2f64, 2f64], 2);
//! let d: (Vector2<f64>, usize) = (vector![3f64, 3f64], 3);
//!
//! let mut kdtree = KdTree::new_static();
//!
//! kdtree.add(&a.0, a.1).unwrap();
//! kdtree.add(&b.0, b.1).unwrap();
//! kdtree.add(&c.0, c.1).unwrap();
//! kdtree.add(&d.0, d.1).unwrap();
//!
//! assert_eq!(kdtree.size(), 4);
//! assert_eq!(
//!     kdtree.nearest(&a.0, 0, &EuclideanNormSquared).unwrap(),
//!     vec![]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 1, &EuclideanNormSquared).unwrap(),
//!     vec![(0f64, &0)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 2, &EuclideanNormSquared).unwrap(),
//!     vec![(0f64, &0), (2f64, &1)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 3, &EuclideanNormSquared).unwrap(),
//!     vec![(0f64, &0), (2f64, &1), (8f64, &2)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 4, &EuclideanNormSquared).unwrap(),
//!     vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&a.0, 5, &EuclideanNormSquared).unwrap(),
//!     vec![(0f64, &0), (2f64, &1), (8f64, &2), (18f64, &3)]
//! );
//! assert_eq!(
//!     kdtree.nearest(&b.0, 4, &EuclideanNormSquared).unwrap(),
//!     vec![(0f64, &1), (2f64, &0), (2f64, &2), (8f64, &3)]
//! );
//! ```

extern crate nalgebra;
extern crate num_traits;

#[cfg(feature = "serialize")]
extern crate serde;

mod heap_element;
pub mod kdtree;
pub mod norm;
mod util;
pub use crate::kdtree::ErrorKind;
pub use crate::kdtree::KdTree;
