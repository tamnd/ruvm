// SPDX-License-Identifier: MIT OR Apache-2.0

//! The unsafe budgets from spec/24-workspace-layout.md.
//!
//! Every crate declares in its manifest how many unsafe blocks, unsafe functions and unsafe impls
//! it may hold, under `[package.metadata.ruvm] unsafe-budget`. This walks each crate's source
//! with syn and fails if the count is over. The count is of syntax, not of lines, so splitting one
//! unsafe block into three is three and merging three into one is one, which is the direction the
//! budget is meant to push.
//!
//! A count under budget is reported too, so that the budget can come down when the code does.

use std::path::Path;

use syn::visit::Visit;

use crate::workspace;

pub(crate) fn check(root: &Path) -> Result<(), String> {
    let packages = workspace::packages(root)?;
    let mut problems = Vec::new();
    let mut total = 0usize;

    for package in &packages {
        let Some(budget) = package.unsafe_budget else {
            if package.name != "xtask" {
                problems.push(format!("{} declares no unsafe-budget", package.name));
            }
            continue;
        };
        let mut sources = Vec::new();
        workspace::rust_files(&package.dir, &mut sources)?;
        let mut count = 0usize;
        for file in &sources {
            let text = std::fs::read_to_string(file)
                .map_err(|e| format!("could not read {}: {e}", file.display()))?;
            count += count_unsafe(&text)
                .map_err(|e| format!("could not parse {}: {e}", file.display()))?;
        }
        total += count;
        let budget = usize::try_from(budget).unwrap_or(usize::MAX);
        if count > budget {
            problems.push(format!(
                "{} holds {count} unsafe items and its budget is {budget}",
                package.name
            ));
        } else if count < budget {
            println!("  {} holds {count} of a budget of {budget}", package.name);
        }
    }

    if problems.is_empty() {
        println!("every crate is within its unsafe budget, {total} unsafe items in all");
        Ok(())
    } else {
        for problem in &problems {
            eprintln!("  {problem}");
        }
        Err(format!("{} crates over their unsafe budget", problems.len()))
    }
}

#[derive(Default)]
struct Counter {
    count: usize,
}

impl<'ast> Visit<'ast> for Counter {
    fn visit_expr_unsafe(&mut self, node: &'ast syn::ExprUnsafe) {
        self.count += 1;
        syn::visit::visit_expr_unsafe(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        if node.sig.unsafety.is_some() {
            self.count += 1;
        }
        syn::visit::visit_item_fn(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        if node.sig.unsafety.is_some() {
            self.count += 1;
        }
        syn::visit::visit_impl_item_fn(self, node);
    }

    fn visit_item_impl(&mut self, node: &'ast syn::ItemImpl) {
        if node.unsafety.is_some() {
            self.count += 1;
        }
        syn::visit::visit_item_impl(self, node);
    }

    fn visit_item_foreign_mod(&mut self, node: &'ast syn::ItemForeignMod) {
        self.count += 1;
        syn::visit::visit_item_foreign_mod(self, node);
    }
}

fn count_unsafe(text: &str) -> syn::Result<usize> {
    let file = syn::parse_file(text)?;
    let mut counter = Counter::default();
    counter.visit_file(&file);
    Ok(counter.count)
}

#[cfg(test)]
mod tests {
    use super::count_unsafe;

    #[test]
    fn blocks_functions_impls_and_extern_blocks_count() {
        let text = r#"
            unsafe fn a() {}
            struct S;
            unsafe impl Send for S {}
            unsafe extern "C" { fn f(); }
            fn b() {
                // SAFETY: a has no preconditions.
                unsafe { a() }
            }
        "#;
        assert_eq!(count_unsafe(text).unwrap(), 4);
    }

    #[test]
    fn safe_code_counts_zero() {
        assert_eq!(count_unsafe("fn main() { let unsafe_name = 1; }").unwrap(), 0);
    }
}
