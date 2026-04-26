//! Monads, adjunctions, and Kleisli composition.
//!
//! A monad on a category C consists of an endofunctor T: C → C together with
//! natural transformations η: Id ⇒ T (unit) and μ: T² ⇒ T (join).
//!
//! # Examples
//!
//! ```
//! use ix_category::monad::{OptionMonad, Monad};
//!
//! // Option is a monad: unit wraps in Some, join flattens Option<Option<T>>
//! let x = OptionMonad::unit(42);
//! assert_eq!(x, Some(42));
//!
//! let nested: Option<Option<i32>> = Some(Some(10));
//! let flat = OptionMonad::join(nested);
//! assert_eq!(flat, Some(10));
//!
//! // Kleisli composition: chain fallible computations
//! let safe_div = |x: i32| -> Option<i32> { if x == 0 { None } else { Some(100 / x) } };
//! let result = OptionMonad::bind(Some(5), safe_div);
//! assert_eq!(result, Some(20));
//!
//! let result = OptionMonad::bind(Some(0), safe_div);
//! assert_eq!(result, None);
//! ```

/// A monad: endofunctor with unit (return) and join (flatten).
///
/// Laws:
/// - `join(unit(m)) = m` (left unit)
/// - `join(map(unit, m)) = m` (right unit)
/// - `join(join(m)) = join(map(join, m))` (associativity)
pub trait Monad {
    type Inner;
    type Wrapped<U>;

    /// Unit (return): wrap a value in the monadic context.
    fn unit(value: Self::Inner) -> Self::Wrapped<Self::Inner>;

    /// Join (flatten): collapse one layer of monadic nesting.
    fn join(wrapped: Self::Wrapped<Self::Wrapped<Self::Inner>>) -> Self::Wrapped<Self::Inner>;

    /// Bind (>>=): apply a function that returns a wrapped value.
    fn bind<F, U>(wrapped: Self::Wrapped<Self::Inner>, f: F) -> Self::Wrapped<U>
    where
        F: FnOnce(Self::Inner) -> Self::Wrapped<U>;
}

/// Option monad implementation.
pub struct OptionMonad;

impl Monad for OptionMonad {
    type Inner = i32;
    type Wrapped<U> = Option<U>;

    fn unit(value: i32) -> Option<i32> {
        Some(value)
    }

    fn join(wrapped: Option<Option<i32>>) -> Option<i32> {
        wrapped.flatten()
    }

    fn bind<F, U>(wrapped: Option<i32>, f: F) -> Option<U>
    where
        F: FnOnce(i32) -> Option<U>,
    {
        wrapped.and_then(f)
    }
}

/// Result monad implementation (with String error).
pub struct ResultMonad;

impl Monad for ResultMonad {
    type Inner = i32;
    type Wrapped<U> = Result<U, String>;

    fn unit(value: i32) -> Result<i32, String> {
        Ok(value)
    }

    fn join(wrapped: Result<Result<i32, String>, String>) -> Result<i32, String> {
        wrapped.and_then(|inner| inner)
    }

    fn bind<F, U>(wrapped: Result<i32, String>, f: F) -> Result<U, String>
    where
        F: FnOnce(i32) -> Result<U, String>,
    {
        wrapped.and_then(f)
    }
}

/// An adjunction F ⊣ G between two functors.
///
/// An adjunction consists of:
/// - Left functor F: C → D
/// - Right functor G: D → C
/// - Unit η: Id_C ⇒ GF
/// - Counit ε: FG ⇒ Id_D
///
/// Such that the triangle identities hold:
/// - εF ∘ Fη = id_F
/// - Gε ∘ ηG = id_G
#[derive(Debug, Clone)]
pub struct Adjunction<L, R> {
    /// Description of the left functor.
    pub left: L,
    /// Description of the right functor.
    pub right: R,
}

/// A concrete adjunction between finite sets: Free ⊣ Forgetful.
///
/// Free functor: takes a set S and produces the free monoid (list) over S.
/// Forgetful functor: takes a monoid and returns its underlying set.
#[derive(Debug, Clone)]
pub struct FreeForgetfulAdj;

impl FreeForgetfulAdj {
    /// Free functor: S → List(S) — wrap elements in singleton lists.
    pub fn free<T: Clone>(elements: &[T]) -> Vec<Vec<T>> {
        elements.iter().map(|e| vec![e.clone()]).collect()
    }

    /// Forgetful functor: List(S) → S — extract the underlying elements.
    pub fn forget<T: Clone>(lists: &[Vec<T>]) -> Vec<T> {
        lists.iter().flat_map(|l| l.iter().cloned()).collect()
    }

    /// Unit: η(x) = [x] — embed an element into the free monoid.
    pub fn unit<T: Clone>(x: &T) -> Vec<T> {
        vec![x.clone()]
    }

    /// Counit: ε(xss) = concat(xss) — flatten the free monoid.
    pub fn counit<T: Clone>(xss: &[Vec<T>]) -> Vec<T> {
        xss.iter().flat_map(|xs| xs.iter().cloned()).collect()
    }
}

/// Kleisli composition for any bind-like operation.
///
/// Given f: A → M<B> and g: B → M<C>, produces h: A → M<C>.
pub fn kleisli_compose<A, B, C, M, F, G>(f: F, g: G) -> impl FnOnce(A) -> M
where
    F: FnOnce(A) -> Option<B>,
    G: FnOnce(B) -> Option<C>,
    M: From<Option<C>>,
{
    move |a| M::from(f(a).and_then(g))
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Option monad ──

    #[test]
    fn test_option_unit() {
        assert_eq!(OptionMonad::unit(42), Some(42));
    }

    #[test]
    fn test_option_join_some_some() {
        assert_eq!(OptionMonad::join(Some(Some(10))), Some(10));
    }

    #[test]
    fn test_option_join_some_none() {
        assert_eq!(OptionMonad::join(Some(None)), None);
    }

    #[test]
    fn test_option_join_none() {
        assert_eq!(OptionMonad::join(None), None);
    }

    #[test]
    fn test_option_bind_some() {
        let result: Option<i32> = OptionMonad::bind(Some(10), |x| Some(x * 2));
        assert_eq!(result, Some(20));
    }

    #[test]
    fn test_option_bind_none_input() {
        let result: Option<i32> = OptionMonad::bind(None, |x| Some(x * 2));
        assert_eq!(result, None);
    }

    #[test]
    fn test_option_bind_none_output() {
        let result: Option<i32> =
            OptionMonad::bind(Some(0), |x| if x == 0 { None } else { Some(100 / x) });
        assert_eq!(result, None);
    }

    #[test]
    fn test_option_left_unit_law() {
        // bind(unit(a), f) = f(a)
        let f = |x: i32| -> Option<i32> { Some(x + 1) };
        let a = 5;
        assert_eq!(OptionMonad::bind(OptionMonad::unit(a), f), f(a));
    }

    #[test]
    fn test_option_right_unit_law() {
        // bind(m, unit) = m
        let m = Some(42);
        let result: Option<i32> = OptionMonad::bind(m, OptionMonad::unit);
        assert_eq!(result, m);
    }

    // ── Result monad ──

    #[test]
    fn test_result_unit() {
        assert_eq!(ResultMonad::unit(7), Ok(7));
    }

    #[test]
    fn test_result_join_ok_ok() {
        assert_eq!(ResultMonad::join(Ok(Ok(5))), Ok(5));
    }

    #[test]
    fn test_result_join_ok_err() {
        assert_eq!(
            ResultMonad::join(Ok(Err("fail".to_string()))),
            Err("fail".to_string())
        );
    }

    #[test]
    fn test_result_bind_ok() {
        let result: Result<i32, String> = ResultMonad::bind(Ok(10), |x| Ok(x * 3));
        assert_eq!(result, Ok(30));
    }

    #[test]
    fn test_result_bind_err() {
        let result: Result<i32, String> = ResultMonad::bind(Err("bad".to_string()), |x| Ok(x * 3));
        assert_eq!(result, Err("bad".to_string()));
    }

    // ── Free ⊣ Forgetful adjunction ──

    #[test]
    fn test_free_functor() {
        let elements = vec![1, 2, 3];
        let free = FreeForgetfulAdj::free(&elements);
        assert_eq!(free, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_forget_functor() {
        let lists = vec![vec![1, 2], vec![3]];
        let forgot = FreeForgetfulAdj::forget(&lists);
        assert_eq!(forgot, vec![1, 2, 3]);
    }

    #[test]
    fn test_unit_adjunction() {
        let x = 42;
        assert_eq!(FreeForgetfulAdj::unit(&x), vec![42]);
    }

    #[test]
    fn test_counit_adjunction() {
        let xss = vec![vec![1, 2], vec![3, 4]];
        assert_eq!(FreeForgetfulAdj::counit(&xss), vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_triangle_identity_counit_free() {
        // ε ∘ F(η) = id_F
        // For element x, F(η)(F(x)) = F([x]) = [[x]]
        // then ε([[x]]) = [x] = F(x) ✓
        let x = 5;
        let fx = vec![x]; // F(x) = [x]
        let f_eta_fx: Vec<Vec<i32>> = fx.iter().map(FreeForgetfulAdj::unit).collect(); // [[x]]
        let result = FreeForgetfulAdj::counit(&f_eta_fx); // [x]
        assert_eq!(result, fx);
    }
}
