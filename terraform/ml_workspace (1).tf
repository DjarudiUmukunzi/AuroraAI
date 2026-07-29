# Phase 2: Azure ML workspace + its required companion resources.
# Azure ML workspaces can't stand alone - they need their own Key Vault,
# Application Insights, and storage account. We give it a dedicated
# storage account (not the ADLS Gen2 one from Phase 1) since Azure ML's
# default storage has different requirements than the Lakehouse.

data "azurerm_client_config" "current" {}

resource "azurerm_application_insights" "ml" {
  name                = "appi-${var.project}-${var.unique_suffix}"
  location            = data.azurerm_resource_group.main.location
  resource_group_name = data.azurerm_resource_group.main.name
  application_type    = "web"
}

resource "azurerm_key_vault" "ml" {
  name                       = "kv-${var.project}-${var.unique_suffix}"
  location                   = data.azurerm_resource_group.main.location
  resource_group_name        = data.azurerm_resource_group.main.name
  tenant_id                  = data.azurerm_client_config.current.tenant_id
  sku_name                   = "standard"
  purge_protection_enabled   = false # keeps teardown simple for a dev/portfolio project
  soft_delete_retention_days = 7
}

resource "azurerm_storage_account" "ml" {
  name                     = "stml${var.project}${var.unique_suffix}"
  resource_group_name      = data.azurerm_resource_group.main.name
  location                 = data.azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version           = "TLS1_2"
}

resource "azurerm_machine_learning_workspace" "main" {
  name                    = "mlw-${var.project}-${var.unique_suffix}"
  location                = data.azurerm_resource_group.main.location
  resource_group_name     = data.azurerm_resource_group.main.name
  application_insights_id = azurerm_application_insights.ml.id
  key_vault_id             = azurerm_key_vault.ml.id
  storage_account_id       = azurerm_storage_account.ml.id
  public_network_access_enabled = true

  identity {
    type = "SystemAssigned"
  }

  tags = {
    project     = var.project
    environment = "dev"
  }
}

# Note: Azure ML automatically creates its own access policy on the linked
# Key Vault when the workspace is created (via its managed identity), so we
# don't need to declare one ourselves here - doing so causes a duplicate-
# resource conflict.

output "ml_workspace_name" {
  value = azurerm_machine_learning_workspace.main.name
}
