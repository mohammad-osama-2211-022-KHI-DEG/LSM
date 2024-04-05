import os
import requests
from requests.auth import HTTPBasicAuth

def get_jwt_token(auth_url, username, password):
    headers = {
        "X-SERVER-TO-SERVER": "true"
    }

    # Make a request to the authentication endpoint with basic authentication
    response = requests.get(auth_url, auth=HTTPBasicAuth(username, password), headers=headers)

    # Check if the request was successful (status code 2xx)
    if response.status_code // 100 == 2:
        # Extract and return the JWT token
        return response.headers['Authorization']
    else:
        # Handle authentication error
        raise ValueError(f"Authentication failed with status code {response.status_code}")
   
def send_data_to_endpoint(data, endpoint, jwt_token):

    jwt = jwt_token

    headers = {
        "Content-Type": "application/json",
        "Authorization": jwt,
        "X-SERVER-TO-SERVER": "true"
    }

    # Make a request to the specified endpoint with the provided data and headers
    response = requests.post(endpoint, json=data, headers=headers)

    # Check if the request was successful (status code 2xx)
    if response.status_code // 100 == 2:
        print("Data sent successfully.")
    elif response.status_code == 401:
        # If unauthorized, refresh the JWT token and retry
        print("JWT token expired. Refreshing and retrying...")
        jwt_token = get_jwt_token()
        #jwt_token = new_jwt_token
        send_data_to_endpoint(data, endpoint, jwt_token)

    else:
        # Handle other errors
        print(f"Error sending data. Status code: {response.status_code}, Response: {response.text}")