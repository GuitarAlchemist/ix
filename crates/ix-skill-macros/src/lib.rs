//! `#[ix_skill]` — annotates a plain Rust function as a registry-discoverable skill.
//!
//! The macro:
//!  1. Parses the function signature via `syn`.
//!  2. Rejects generics, `self` params, and functions with no return type.
//!  3. Emits an `__ix_skill_adapter_<name>` fn that decodes `&[Value]` into
//!     the native arg types via [`FromValue`] and re-encodes the return via
//!     [`IntoValue`].
//!  4. Emits a `static` [`SkillDescriptor`] registered to `IX_SKILLS` via
//!     `#[linkme::distributed_slice]`.
//!
//! # Attribute syntax
//!
//! ```ignore
//! #[ix_skill(domain = "supervised", governance = "empirical,deterministic")]
//! pub fn fit(x: IxMatrix, y: IxVector, lr: f64, epochs: usize) -> f64 { ... }
//! ```
//!
//! Supported keys:
//! - `domain` — domain tag, e.g. `"supervised"`
//! - `governance` — comma-separated governance tags
//! - `name` — override the auto-derived dotted name
//! - `schema_fn` — path to a user-defined `fn() -> serde_json::Value` that
//!   returns a hand-written JSON schema (used when auto-derived schema is
//!   insufficient, e.g. for composite MCP handlers)
//!
//! # Requirements on the annotated fn
//! - No generics (compile-error if generics present).
//! - No `self` / `&self` (free function only).
//! - Every argument type must impl `ix_types::FromValue`.
//! - Return type must impl `ix_types::IntoValue`, or be `Result<T, E>` where
//!   `T: IntoValue` and `E: core::fmt::Display`.
//!
//! [`FromValue`]: ix_types::FromValue
//! [`IntoValue`]: ix_types::IntoValue
//! [`SkillDescriptor`]: ix_registry::SkillDescriptor

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parser, parse_macro_input, parse_str, punctuated::Punctuated, Expr, ExprLit, FnArg,
    ItemFn, Lit, Meta, Pat, Path, PathArguments, ReturnType, Token, Type,
};

#[proc_macro_attribute]
pub fn ix_skill(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let parser = Punctuated::<Meta, Token![,]>::parse_terminated;
    let metas = match parser.parse(attr) {
        Ok(m) => m,
        Err(e) => return e.to_compile_error().into(),
    };

    // --- Parse attribute args -------------------------------------------------
    let mut domain = String::from("general");
    let mut governance: Vec<String> = Vec::new();
    let mut name_override: Option<String> = None;
    let mut schema_fn_path: Option<Path> = None;
    for m in &metas {
        if let Meta::NameValue(nv) = m {
            let key = nv
                .path
                .get_ident()
                .map(|i| i.to_string())
                .unwrap_or_default();
            if let Expr::Lit(ExprLit {
                lit: Lit::Str(s), ..
            }) = &nv.value
            {
                match key.as_str() {
                    "domain" => domain = s.value(),
                    "governance" => {
                        governance = s
                            .value()
                            .split(',')
                            .map(|t| t.trim().to_string())
                            .filter(|t| !t.is_empty())
                            .collect();
                    }
                    "name" => name_override = Some(s.value()),
                    "schema_fn" => match parse_str::<Path>(&s.value()) {
                        Ok(p) => schema_fn_path = Some(p),
                        Err(e) => return e.to_compile_error().into(),
                    },
                    _ => {}
                }
            }
        }
    }

    // --- Reject generics & self ----------------------------------------------
    if !func.sig.generics.params.is_empty() {
        return syn::Error::new_spanned(
            &func.sig.generics,
            "#[ix_skill] does not support generic functions — monomorphize at a wrapper",
        )
        .to_compile_error()
        .into();
    }
    if let Some(FnArg::Receiver(r)) = func.sig.inputs.first() {
        return syn::Error::new_spanned(r, "#[ix_skill] only applies to free functions")
            .to_compile_error()
            .into();
    }
    let ReturnType::Type(_, ret_ty_box) = &func.sig.output else {
        return syn::Error::new_spanned(&func.sig, "#[ix_skill] requires an explicit return type")
            .to_compile_error()
            .into();
    };
    let ret_ty = &**ret_ty_box;

    // --- Walk params ---------------------------------------------------------
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();
    let crate_name = std::env::var("CARGO_PKG_NAME").unwrap_or_else(|_| "unknown".into());
    let domain_for_name = domain.clone();
    let skill_name = name_override.unwrap_or_else(|| {
        let crate_short = crate_name
            .strip_prefix("ix-")
            .unwrap_or(crate_name.as_str());
        format!("{crate_short}.{domain_for_name}.{fn_name_str}")
    });

    // Collect arg socket decls + decoders.
    let mut socket_decls = Vec::new();
    let mut arg_decoders = Vec::new();
    let mut arg_idents = Vec::new();
    for (i, arg) in func.sig.inputs.iter().enumerate() {
        let FnArg::Typed(pt) = arg else {
            return syn::Error::new_spanned(arg, "unexpected receiver")
                .to_compile_error()
                .into();
        };
        let ident = match &*pt.pat {
            Pat::Ident(pi) => pi.ident.clone(),
            _ => format_ident!("__arg{}", i),
        };
        let ty = &*pt.ty;
        let name_str = ident.to_string();
        socket_decls.push(quote! {
            ::ix_registry::Socket {
                name: #name_str,
                ty: <#ty as ::ix_types::FromValue>::SOCKET,
                optional: false,
                doc: "",
            }
        });
        arg_decoders.push(quote! {
            let #ident = <#ty as ::ix_types::FromValue>::from_value(&args[#i])
                .map_err(|e| ::ix_registry::SkillError::Type { arg_index: #i, source: e })?;
        });
        arg_idents.push(ident);
    }
    let n_inputs = arg_idents.len();

    // --- Detect Result<T, E> return ------------------------------------------
    let (inner_ret_ty, is_result) = detect_result_inner(ret_ty);

    let output_socket = quote! {
        ::ix_registry::Socket {
            name: "out",
            ty: <#inner_ret_ty as ::ix_types::IntoValue>::SOCKET,
            optional: false,
            doc: "",
        }
    };

    let call_and_wrap = if is_result {
        quote! {
            let out: #inner_ret_ty = #fn_name(#(#arg_idents),*)
                .map_err(|e| ::ix_registry::SkillError::Exec(format!("{}", e)))?;
            Ok(<#inner_ret_ty as ::ix_types::IntoValue>::into_value(out))
        }
    } else {
        quote! {
            let out: #inner_ret_ty = #fn_name(#(#arg_idents),*);
            Ok(<#inner_ret_ty as ::ix_types::IntoValue>::into_value(out))
        }
    };

    // --- Doc comment harvest -------------------------------------------------
    let doc_text = func
        .attrs
        .iter()
        .filter_map(|a| {
            if a.path().is_ident("doc") {
                if let Meta::NameValue(nv) = &a.meta {
                    if let Expr::Lit(ExprLit {
                        lit: Lit::Str(s), ..
                    }) = &nv.value
                    {
                        return Some(s.value());
                    }
                }
            }
            None
        })
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();

    // --- Identifiers for generated items -------------------------------------
    let adapter_ident = format_ident!("__ix_skill_adapter_{}", fn_name);
    let schema_ident = format_ident!("__ix_skill_schema_{}", fn_name);
    let static_ident = format_ident!("__IX_SKILL_{}", fn_name.to_string().to_uppercase());
    let crate_lit = crate_name.clone();
    let domain_lit = domain.clone();
    let gov_lits: Vec<_> = governance.iter().map(|g| quote!(#g)).collect();
    let resolved_schema_fn = match &schema_fn_path {
        Some(p) => quote!(#p),
        None => quote!(#schema_ident),
    };

    let expanded = quote! {
        #func

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #adapter_ident(args: &[::ix_types::Value])
            -> ::core::result::Result<::ix_types::Value, ::ix_registry::SkillError>
        {
            if args.len() != #n_inputs {
                return Err(::ix_registry::SkillError::Arity {
                    expected: #n_inputs,
                    actual: args.len(),
                });
            }
            #(#arg_decoders)*
            #call_and_wrap
        }

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #schema_ident() -> ::serde_json::Value {
            ::serde_json::json!({
                "name": #skill_name,
                "arity": #n_inputs,
            })
        }

        // Resolve which schema function the descriptor points at. If the user
        // supplied `schema_fn = "path::to::fn"` we use that; otherwise we use
        // the auto-generated stub above.

        #[::linkme::distributed_slice(::ix_registry::IX_SKILLS)]
        #[linkme(crate = ::ix_registry::linkme)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        static #static_ident: ::ix_registry::SkillDescriptor =
            ::ix_registry::SkillDescriptor {
                name: #skill_name,
                domain: #domain_lit,
                crate_name: #crate_lit,
                doc: #doc_text,
                inputs: &[ #(#socket_decls),* ],
                outputs: &[ #output_socket ],
                governance_tags: &[ #(#gov_lits),* ],
                json_schema: #resolved_schema_fn,
                fn_ptr: #adapter_ident,
            };
    };

    expanded.into()
}

/// If `ty` is `Result<T, E>`, returns `(T, true)`. Otherwise `(ty, false)`.
fn detect_result_inner(ty: &Type) -> (Type, bool) {
    if let Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Result" {
                if let PathArguments::AngleBracketed(ab) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = ab.args.first() {
                        return (inner.clone(), true);
                    }
                }
            }
        }
    }
    (ty.clone(), false)
}
