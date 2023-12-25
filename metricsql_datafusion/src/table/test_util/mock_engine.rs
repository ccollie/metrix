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

use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::table::engine::{EngineContext, TableEngine};
use crate::table::error::Result;
use crate::table::metadata::TableId;
use crate::table::requests::{CreateTableRequest, DropTableRequest, OpenTableRequest};
use crate::table::table::TableRef;
use crate::table::test_util::empty_table::EmptyTable;

#[derive(Default)]
pub struct MockTableEngine {
    tables: Mutex<HashMap<(String, String, String), TableRef>>,
}

impl MockTableEngine {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl TableEngine for MockTableEngine {
    fn name(&self) -> &str {
        "MockTableEngine"
    }

    async fn create_table(
        &self,
        _ctx: &EngineContext,
        request: CreateTableRequest,
    ) -> Result<TableRef> {
        let catalog_name = request.catalog_name.clone();
        let schema_name = request.schema_name.clone();
        let table_name = request.table_name.clone();

        let table_ref = EmptyTable::table(request);

        let _ = self
            .tables
            .lock()
            .await
            .insert((catalog_name, schema_name, table_name), table_ref.clone());
        Ok(table_ref)
    }

    async fn open_table(
        &self,
        _ctx: &EngineContext,
        request: OpenTableRequest,
    ) -> Result<Option<TableRef>> {
        let catalog_name = request.catalog_name;
        let schema_name = request.schema_name;
        let table_name = request.table_name;

        let res = self
            .tables
            .lock()
            .await
            .get(&(catalog_name, schema_name, table_name))
            .cloned();

        Ok(res)
    }

    fn get_table(&self, _ctx: &EngineContext, _table_id: TableId) -> Result<Option<TableRef>> {
        unimplemented!()
    }

    fn table_exists(&self, _ctx: &EngineContext, _table_id: TableId) -> bool {
        unimplemented!()
    }

    async fn drop_table(&self, _ctx: &EngineContext, _request: DropTableRequest) -> Result<bool> {
        unimplemented!()
    }

    async fn close(&self) -> Result<()> {
        Ok(())
    }
}