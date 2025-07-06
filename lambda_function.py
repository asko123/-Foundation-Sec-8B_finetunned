import json
from main import vault
from main import jenkins
from main import utils
import ssl
from urllib import error as urllib_error
from http import client as http_client
import socket

def main(event, context):
    print("Starting Jenkins verification process")
    lambda_account_id = context.invoked_function_arn.split(":")[4]

    # Writes CA chain file to a file
    try:
        utils.get_ca_cert('/tmp/ca-chain.cer')
        print('Certificate chain retrieved successfully')
    except (IOError, OSError):
        print('Failed to write certificate chain to filesystem')
        return {
            'statusCode': 503,
            'body': json.dumps({
                'error': 'System configuration error'
            })
        }
    except Exception:
        print('Failed to retrieve certificate chain')
        return {
            'statusCode': 502,
            'body': json.dumps({
                'error': 'Certificate configuration error'
            })
        }

    # Get credentials with detailed error handling
    try:
        aws_creds = vault.get_aws_creds(lambda_account_id)
        jenkins_token = vault.get_jenkins_token(lambda_account_id)
    except (ConnectionError, socket.gaierror) as e:
        error_msg = f'Network error connecting to vault: {str(e)}'
        print(error_msg)
        return {
            'statusCode': 503,
            'body': json.dumps({
                'error': 'Authentication service unavailable',
                'details': 'Network connectivity issue'
            })
        }
    except socket.timeout:
        print('Vault connection timed out')
        return {
            'statusCode': 504,
            'body': json.dumps({
                'error': 'Authentication service timeout',
                'details': 'Connection to vault timed out'
            })
        }
    except (ValueError, TypeError) as e:
        error_msg = f'Invalid credentials format: {str(e)}'
        print(error_msg)
        return {
            'statusCode': 400,
            'body': json.dumps({
                'error': 'Invalid credentials format',
                'details': 'Malformed credential data received'
            })
        }
    except KeyError as e:
        error_msg = f'Missing required credential: {str(e)}'
        print(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Missing credentials',
                'details': 'Required credential data not found'
            })
        }
    except PermissionError:
        print('Permission denied accessing credentials')
        return {
            'statusCode': 403,
            'body': json.dumps({
                'error': 'Permission denied',
                'details': 'Insufficient permissions to access credentials'
            })
        }
    except FileNotFoundError:
        print('Credential file or configuration not found')
        return {
            'statusCode': 404,
            'body': json.dumps({
                'error': 'Configuration not found',
                'details': 'Required credential configuration missing'
            })
        }
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f'Unexpected error retrieving credentials: {error_type}'
        print(error_msg)
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Authentication configuration error',
                'details': f'Unexpected error: {error_type}'
            })
        }

    # Call Jenkins
    try:
        return jenkins.call_jenkins_verify(
            aws_creds=aws_creds,
            jenkins_token=jenkins_token
        )
    except ssl.SSLError:
        print('SSL verification failed')
        return {
            'statusCode': 495,  # SSL Certificate Error
            'body': json.dumps({
                'error': 'SSL verification failed'
            })
        }
    except ssl.CertificateError:
        print('Certificate verification failed')
        return {
            'statusCode': 495,  # SSL Certificate Error
            'body': json.dumps({
                'error': 'Certificate verification failed'
            })
        }
    except urllib_error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError):
            print('Failed to connect to Jenkins')
            return {
                'statusCode': 503,
                'body': json.dumps({
                    'error': 'Jenkins service unavailable'
                })
            }
        elif isinstance(e.reason, TimeoutError):
            print('Jenkins request timed out')
            return {
                'statusCode': 504,
                'body': json.dumps({
                    'error': 'Jenkins request timed out'
                })
            }
        else:
            print('Failed to reach Jenkins')
            return {
                'statusCode': 502,
                'body': json.dumps({
                    'error': 'Jenkins service unreachable'
                })
            }
    except urllib_error.HTTPError as e:
        print(f'Jenkins returned HTTP {e.code}')
        return {
            'statusCode': e.code,
            'body': json.dumps({
                'error': 'Jenkins request failed'
            })
        }
    except http_client.HTTPException:
        print('Invalid HTTP response from Jenkins')
        return {
            'statusCode': 502,
            'body': json.dumps({
                'error': 'Invalid response from Jenkins'
            })
        }
    except Exception:
        print('Jenkins verification failed')
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error'
            })
        } 