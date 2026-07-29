variable "resource_group_name" {
  description = "Existing resource group from Phase 0 (Foundry setup)"
  type        = string
  default     = "rg-auroraai-dev"
}

variable "location" {
  description = "Azure region - must match your resource group's region"
  type        = string
  default     = "eastus"
}

variable "project" {
  description = "Short project name used in resource naming"
  type        = string
  default     = "auroraai"
}

variable "unique_suffix" {
  description = "Unique suffix for globally-unique resource names (e.g. your name/initials) - storage accounts and Databricks workspaces must be globally unique"
  type        = string
}
