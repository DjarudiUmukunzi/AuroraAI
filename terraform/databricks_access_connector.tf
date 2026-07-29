# Lets Databricks (including Serverless compute, which can't use raw
# storage account keys) authenticate to ADLS Gen2 via a managed identity
# instead. This is the modern, Unity-Catalog-native replacement for the
# secret-scope/account-key approach.

resource "azurerm_databricks_access_connector" "main" {
  name                = "dbx-connector-${var.project}-${var.unique_suffix}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location

  identity {
    type = "SystemAssigned"
  }

  tags = {
    project     = var.project
    environment = "dev"
  }
}

# Let that connector's managed identity read/write the data lake
resource "azurerm_role_assignment" "dbx_connector_storage" {
  scope                = azurerm_storage_account.datalake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_databricks_access_connector.main.identity[0].principal_id
}

output "databricks_access_connector_id" {
  description = "Paste this into Databricks Catalog Explorer when creating the Storage Credential"
  value       = azurerm_databricks_access_connector.main.id
}
