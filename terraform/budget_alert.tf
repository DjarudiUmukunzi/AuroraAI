# Phase 4 cost optimization: a real budget with alert thresholds, so
# spend gets flagged before it becomes a surprise rather than being
# discovered after the fact. $20/month is generous headroom for this
# project's actual usage (mostly $0 - Free-tier AI Search, gpt-5-mini/
# text-embedding-3-small token costs measured in cents, ADLS/ADF at
# near-zero dev volume) - the point is having the alert wired up and
# working, not that we expect to hit it.

data "azurerm_client_config" "budget" {}

resource "azurerm_consumption_budget_resource_group" "main" {
  name              = "budget-${var.project}-monthly"
  resource_group_id = data.azurerm_resource_group.main.id

  amount     = 20
  time_grain = "Monthly"

  time_period {
    start_date = "2026-08-01T00:00:00Z"
    # No end_date - an open-ended recurring monthly budget
  }

  notification {
    enabled        = true
    threshold      = 50
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = ["djarudimukunzi@outlook.com"]
     
  }

  notification {
    enabled        = true
    threshold      = 80
    operator       = "GreaterThan"
    threshold_type = "Actual"
    contact_emails = ["djarudimukunzi@outlook.com"]
  }

  notification {
    enabled        = true
    threshold      = 100
    operator       = "GreaterThan"
    threshold_type = "Forecasted"
    contact_emails = ["djarudimukunzi@outlook.com"]
  }
}
