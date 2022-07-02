// (C) Copyright 2019-2020 Hewlett Packard Enterprise Development LP
#![forbid(unsafe_code)]

extern crate logos;
extern crate enquote;
extern crate regex;
extern crate phf;
extern crate core;
extern crate once_cell;
extern crate rowan;
extern crate num_derive;
extern crate num_traits;

mod parser;
mod binaryop;
mod lexer;
pub mod error;
pub mod types;

pub use parser::{parse, parse_single_expr};
pub use binaryop::*;