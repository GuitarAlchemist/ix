//! Core category theory abstractions.
//!
//! Defines traits for categories, functors, natural transformations,
//! and monoidal categories. These map naturally to Rust's type system:
//! a category's morphisms are like trait implementations.
//!
//! # Examples
//!
//! ```
//! use machin_category::core::Category;
//!
//! // A simple category where objects are types and morphisms are functions
//! struct FnCategory;
//!
//! impl Category for FnCategory {
//!     type Obj = &'static str;
//!     type Mor = fn(f64) -> f64;
//!
//!     fn id(_obj: &Self::Obj) -> Self::Mor {
//!         |x| x
//!     }
//!
//!     fn compose(f: &Self::Mor, g: &Self::Mor) -> Self::Mor {
//!         // Note: compose(f, g) = g ∘ f (apply f first, then g)
//!         // We can't compose function pointers directly, so return identity
//!         // as a demonstration. Real composition needs closures.
//!         |x| x
//!     }
//!
//!     fn dom(_f: &Self::Mor) -> Self::Obj { "R" }
//!     fn cod(_f: &Self::Mor) -> Self::Obj { "R" }
//! }
//!
//! let id = FnCategory::id(&"R");
//! assert_eq!(id(42.0), 42.0);
//! ```

/// A category consists of objects, morphisms, identity, and composition.
///
/// Laws (not enforced at compile time):
/// - `compose(id(a), f) = f` (left identity)
/// - `compose(f, id(b)) = f` (right identity)
/// - `compose(compose(f, g), h) = compose(f, compose(g, h))` (associativity)
pub trait Category {
    /// The type of objects in this category.
    type Obj: Clone + PartialEq;
    /// The type of morphisms (arrows) in this category.
    type Mor: Clone;

    /// Identity morphism for an object.
    fn id(obj: &Self::Obj) -> Self::Mor;

    /// Compose two morphisms: `compose(f: A→B, g: B→C) = g∘f: A→C`.
    fn compose(f: &Self::Mor, g: &Self::Mor) -> Self::Mor;

    /// Domain (source) of a morphism.
    fn dom(f: &Self::Mor) -> Self::Obj;

    /// Codomain (target) of a morphism.
    fn cod(f: &Self::Mor) -> Self::Obj;
}

/// A functor F: C → D maps objects and morphisms between categories.
///
/// Laws:
/// - `map_mor(id_C(a)) = id_D(map_obj(a))` (preserves identity)
/// - `map_mor(compose(f, g)) = compose(map_mor(f), map_mor(g))` (preserves composition)
pub trait Functor<C: Category, D: Category> {
    /// Map an object from C to D.
    fn map_obj(obj: &C::Obj) -> D::Obj;

    /// Map a morphism from C to D.
    fn map_mor(mor: &C::Mor) -> D::Mor;
}

/// A natural transformation η: F ⇒ G between two functors F, G: C → D.
///
/// For each object A in C, provides a morphism η_A: F(A) → G(A) in D,
/// such that for every morphism f: A → B in C:
///   G(f) ∘ η_A = η_B ∘ F(f) (naturality square commutes)
pub trait NaturalTransformation<C: Category, D: Category> {
    /// The component morphism η_A for object A.
    fn component(obj: &C::Obj) -> D::Mor;
}

/// A monoidal category has a tensor product and unit object.
///
/// Laws (coherence conditions):
/// - `tensor(tensor(a, b), c) ≅ tensor(a, tensor(b, c))` (associativity up to isomorphism)
/// - `tensor(unit(), a) ≅ a` and `tensor(a, unit()) ≅ a` (unit laws)
pub trait Monoidal: Category {
    /// Tensor product of two objects.
    fn tensor_obj(a: &Self::Obj, b: &Self::Obj) -> Self::Obj;

    /// Tensor product of two morphisms.
    fn tensor_mor(f: &Self::Mor, g: &Self::Mor) -> Self::Mor;

    /// The unit object (identity for tensor).
    fn unit() -> Self::Obj;
}

/// Compose two functors: G ∘ F: A → C given F: A → B and G: B → C.
pub struct ComposedFunctor<F, G, B> {
    _f: std::marker::PhantomData<F>,
    _g: std::marker::PhantomData<G>,
    _b: std::marker::PhantomData<B>,
}

impl<A, B, C, F, G> Functor<A, C> for ComposedFunctor<F, G, B>
where
    A: Category,
    B: Category,
    C: Category,
    F: Functor<A, B>,
    G: Functor<B, C>,
{
    fn map_obj(obj: &A::Obj) -> C::Obj {
        G::map_obj(&F::map_obj(obj))
    }

    fn map_mor(mor: &A::Mor) -> C::Mor {
        G::map_mor(&F::map_mor(mor))
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // A simple discrete category: objects are integers, only identity morphisms
    struct DiscreteCategory;

    #[derive(Clone, PartialEq, Debug)]
    struct DiscreteObj(i32);

    #[derive(Clone, Debug)]
    struct DiscreteMor {
        source: i32,
    }

    impl Category for DiscreteCategory {
        type Obj = DiscreteObj;
        type Mor = DiscreteMor;

        fn id(obj: &Self::Obj) -> Self::Mor {
            DiscreteMor { source: obj.0 }
        }

        fn compose(f: &Self::Mor, _g: &Self::Mor) -> Self::Mor {
            // In a discrete category, the only composable pair is id ∘ id = id
            f.clone()
        }

        fn dom(f: &Self::Mor) -> Self::Obj {
            DiscreteObj(f.source)
        }

        fn cod(f: &Self::Mor) -> Self::Obj {
            DiscreteObj(f.source)
        }
    }

    #[test]
    fn test_discrete_category_identity() {
        let obj = DiscreteObj(42);
        let id = DiscreteCategory::id(&obj);
        assert_eq!(DiscreteCategory::dom(&id), obj);
        assert_eq!(DiscreteCategory::cod(&id), obj);
    }

    #[test]
    fn test_discrete_category_compose() {
        let obj = DiscreteObj(7);
        let id = DiscreteCategory::id(&obj);
        let composed = DiscreteCategory::compose(&id, &id);
        assert_eq!(DiscreteCategory::dom(&composed), obj);
    }

    // A category of sets with functions as morphisms (simplified)
    struct SetCategory;

    #[derive(Clone, PartialEq, Debug)]
    struct SetObj(String);

    #[derive(Clone)]
    #[allow(dead_code)]
    struct SetMor {
        source: String,
        target: String,
        /// Mapping as list of (input_idx, output_idx) pairs
        mapping: Vec<(usize, usize)>,
    }

    impl Category for SetCategory {
        type Obj = SetObj;
        type Mor = SetMor;

        fn id(obj: &Self::Obj) -> Self::Mor {
            SetMor {
                source: obj.0.clone(),
                target: obj.0.clone(),
                mapping: vec![],
            }
        }

        fn compose(f: &Self::Mor, g: &Self::Mor) -> Self::Mor {
            SetMor {
                source: f.source.clone(),
                target: g.target.clone(),
                mapping: vec![], // simplified
            }
        }

        fn dom(f: &Self::Mor) -> Self::Obj {
            SetObj(f.source.clone())
        }

        fn cod(f: &Self::Mor) -> Self::Obj {
            SetObj(f.target.clone())
        }
    }

    #[test]
    fn test_set_category_identity_law() {
        let a = SetObj("A".to_string());
        let id_a = SetCategory::id(&a);
        assert_eq!(SetCategory::dom(&id_a), a);
        assert_eq!(SetCategory::cod(&id_a), a);
    }

    #[test]
    fn test_set_category_compose_domains() {
        let a = SetObj("A".to_string());
        let f = SetMor {
            source: "A".into(),
            target: "B".into(),
            mapping: vec![],
        };
        let g = SetMor {
            source: "B".into(),
            target: "C".into(),
            mapping: vec![],
        };
        let gf = SetCategory::compose(&f, &g);
        assert_eq!(SetCategory::dom(&gf), a);
        assert_eq!(SetCategory::cod(&gf).0, "C");
    }

    // Test identity functor
    struct IdentityFunctor;

    impl Functor<DiscreteCategory, DiscreteCategory> for IdentityFunctor {
        fn map_obj(obj: &DiscreteObj) -> DiscreteObj {
            obj.clone()
        }

        fn map_mor(mor: &DiscreteMor) -> DiscreteMor {
            mor.clone()
        }
    }

    #[test]
    fn test_identity_functor() {
        let obj = DiscreteObj(5);
        let mapped = IdentityFunctor::map_obj(&obj);
        assert_eq!(mapped, obj);
    }

    #[test]
    fn test_identity_functor_preserves_identity() {
        let obj = DiscreteObj(5);
        let id = DiscreteCategory::id(&obj);
        let mapped_id = IdentityFunctor::map_mor(&id);
        let id_of_mapped = DiscreteCategory::id(&IdentityFunctor::map_obj(&obj));
        assert_eq!(
            DiscreteCategory::dom(&mapped_id),
            DiscreteCategory::dom(&id_of_mapped)
        );
    }

    // Test composed functor
    type DoubleIdentity = ComposedFunctor<IdentityFunctor, IdentityFunctor, DiscreteCategory>;

    #[test]
    fn test_composed_functor() {
        let obj = DiscreteObj(10);
        let mapped = <DoubleIdentity as Functor<DiscreteCategory, DiscreteCategory>>::map_obj(&obj);
        assert_eq!(mapped, obj);
    }
}
