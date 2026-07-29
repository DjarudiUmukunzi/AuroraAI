output "storage_account_name" {
  value = azurerm_storage_account.datalake.name
}

output "adls_primary_dfs_endpoint" {
  description = "The abfss:// endpoint used by Databricks/Delta Lake to address this storage account"
  value       = azurerm_storage_account.datalake.primary_dfs_endpoint
}

output "data_factory_name" {
  value = azurerm_data_factory.adf.name
}

output "databricks_workspace_url" {
  value = azurerm_databricks_workspace.dbx.workspace_url
}
