// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow_schema::{DataType, Field, Schema, SchemaRef};

use crate::catalog::consts::MITO_ENGINE;

const UNKNOWN: &str = "unknown";
const COLUMN_PRIVILEGES: &str = "column_privileges";
const COLUMN_STATISTICS: &str = "column_statistics";
const ENGINES: &str = "engines";


/// Find the schema and columns by the table_name, only valid for memory tables.
/// Safety: the user MUST ensure the table schema exists, panic otherwise.
pub fn get_schema_columns(table_name: &str) -> (SchemaRef, Vec<ArrayRef>) {
    let (column_schemas, columns): (_, Vec<ArrayRef>) = match table_name {
        COLUMN_PRIVILEGES => (
            string_columns(&[
                "GRANTEE",
                "TABLE_CATALOG",
                "TABLE_SCHEMA",
                "TABLE_NAME",
                "COLUMN_NAME",
                "PRIVILEGE_TYPE",
                "IS_GRANTABLE",
            ]),
            vec![],
        ),

        COLUMN_STATISTICS => (
            string_columns(&[
                "SCHEMA_NAME",
                "TABLE_NAME",
                "COLUMN_NAME",
                // TODO(dennis): It must be a JSON type, but we don't support it yet
                "HISTOGRAM",
            ]),
            vec![],
        ),

        ENGINES => (
            string_columns(&[
                "ENGINE",
                "SUPPORT",
                "COMMENT",
                "TRANSACTIONS",
                "XA",
                "SAVEPOINTS",
            ]),
            vec![
                Arc::new(StringArray::from(vec![MITO_ENGINE])),
                Arc::new(StringArray::from(vec!["DEFAULT"])),
                Arc::new(StringArray::from(vec![
                    "Storage query_engine for time-series data",
                ])),
                Arc::new(StringArray::from(vec!["NO"])),
                Arc::new(StringArray::from(vec!["NO"])),
                Arc::new(StringArray::from(vec!["NO"])),
            ],
        ),

        _ => unreachable!("Unknown table in information_schema: {}", table_name),
    };

    (Arc::new(Schema::new(column_schemas)), columns)
}

fn string_columns(names: &[&'static str]) -> Vec<Field> {
    names.iter().map(|name| string_column(name)).collect()
}

fn string_column(name: &str) -> Field {
    Field::new(
        str::to_lowercase(name),
        DataType::Utf8,
        false,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_columns() {
        let columns = ["a", "b", "c"];
        let column_schemas = string_columns(&columns);

        assert_eq!(3, column_schemas.len());
        for (i, name) in columns.iter().enumerate() {
            let cs = column_schemas.get(i).unwrap();

            assert_eq!(*name, cs.name());
            assert_eq!(DataType::Utf8, cs.data_type());
        }
    }
}
