# Reference the existing resource group created in Phase 0 (Portal setup) -
# we don't recreate it here, just point Terraform at it.
data "azurerm_resource_group" "main" {
  name = var.resource_group_name
}

# --- ADLS Gen2 storage account ---
# is_hns_enabled = true is what makes this "Data Lake Storage Gen2" rather
# than plain Blob storage - it enables the hierarchical namespace that
# Databricks/Delta Lake expect.
resource "azurerm_storage_account" "datalake" {
  name                     = "st${var.project}${var.unique_suffix}"
  resource_group_name      = data.azurerm_resource_group.main.name
  location                 = data.azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS" # cheapest replication tier - fine for a dev/portfolio project
  account_kind             = "StorageV2"
  is_hns_enabled           = true
  min_tls_version           = "TLS1_2"

  tags = {
    project     = var.project
    environment = "dev"
  }
}

# Medallion architecture: bronze (raw), silver (cleaned), gold (aggregated/model-ready)
resource "azurerm_storage_container" "bronze" {
  name                  = "bronze"
  storage_account_id    = azurerm_storage_account.datalake.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "silver" {
  name                  = "silver"
  storage_account_id    = azurerm_storage_account.datalake.id
  container_access_type = "private"
}

resource "azurerm_storage_container" "gold" {
  name                  = "gold"
  storage_account_id    = azurerm_storage_account.datalake.id
  container_access_type = "private"
}

# --- Azure Data Factory ---
# Pipelines inside ADF (built in the ADF Studio UI, not Terraform - Terraform
# just provisions the factory itself) will pull NOAA/NASA data and land it
# in the bronze container.
resource "azurerm_data_factory" "adf" {
  name                = "adf-${var.project}-${var.unique_suffix}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name

  identity {
    type = "SystemAssigned"
  }

  tags = {
    project     = var.project
    environment = "dev"
  }
}

# Let Data Factory's managed identity read/write the data lake
resource "azurerm_role_assignment" "adf_storage_contributor" {
  scope                = azurerm_storage_account.datalake.id
  role_definition_name = "Storage Blob Data Contributor"
  principal_id         = azurerm_data_factory.adf.identity[0].principal_id
}

# --- Databricks workspace ---
# Note: Azure retired the Standard SKU (mid-2026) - Premium is now the only
# option for new workspaces. Premium adds per-DBU cost on top of compute,
# but there's no charge for the workspace shell itself - cost only kicks in
# when you actually run a cluster, so this doesn't change your idle cost.
resource "azurerm_databricks_workspace" "dbx" {
  name                = "dbx-${var.project}-${var.unique_suffix}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  sku                 = "premium"

  tags = {
    project     = var.project
    environment = "dev"
  }
}
