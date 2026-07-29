# Phase 3: Azure AI Search backs the Research Agent's RAG pipeline over
# NOAA/NASA bulletins. Free tier: $0/month, 50MB storage, 3 indexes,
# supports vector + semantic + hybrid search - plenty for this project's
# scale (bulletins are short text documents, not a huge corpus).
# Note: only one Free-tier search service is allowed per subscription.

resource "azurerm_search_service" "main" {
  name                = "srch-${var.project}-${var.unique_suffix}"
  resource_group_name = data.azurerm_resource_group.main.name
  location            = data.azurerm_resource_group.main.location
  sku                 = "free"

  tags = {
    project     = var.project
    environment = "dev"
  }
}

output "search_service_name" {
  value = azurerm_search_service.main.name
}

output "search_service_endpoint" {
  value = "https://${azurerm_search_service.main.name}.search.windows.net"
}
