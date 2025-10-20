# Terraform stub for AWS (extend as needed)
terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}
provider "aws" {
  region = var.region
}
# Example: S3 bucket for artifacts
resource "aws_s3_bucket" "artifacts" {
  bucket = var.bucket_name
  force_destroy = true
}
output "bucket_name" { value = aws_s3_bucket.artifacts.bucket }
