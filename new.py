from qdrant_client import QdrantClient



client = QdrantClient(

    url="https://2468b96d-f0ae-4d97-a440-74b23e10aa08.us-east-1-1.aws.cloud.qdrant.io",

    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzgxMjY3MDU3LCJzdWJqZWN0IjoiYXBpLWtleTowNmE0ODE4My00ZTE5LTQyNTEtOGQ3MC1hYzU0OTNkNmYzMWIifQ.1S4sS7Z8z4iBfQ_IJWoOQb_NN60d2l8SLH2bWCCIzHM"

)



try:

    print(client.get_collections())

    print("Connected successfully")

except Exception as e:

    print(e)