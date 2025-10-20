// Minimal Azure Bicep stub: Storage account for artifacts
param location string = resourceGroup().location
param storageName string = 'llmops${uniqueString(resourceGroup().id)}'

resource stg 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageName
  location: location
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

output storageEndpoint string = stg.properties.primaryEndpoints.blob
