# aws

from opensearchpy import OpenSearch

OPENSEARCH_URL = "https://learn-e779669-os-9200.tale-sandbox.dev.aws.jpmchase.net"

client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=False
)

print("Connecting from Python...")

try:
    info = client.info()
    print("✅ Python connected!")
    print(info)
except Exception as e:
    print("❌ Python connection failed:")
    print(e)
