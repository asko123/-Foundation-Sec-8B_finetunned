import json
import pytest
import ssl
from urllib import error as urllib_error
from http import client as http_client
import socket
import sys
from unittest.mock import MagicMock

# Mock the main module and its submodules before importing lambda_function
mock_main = MagicMock()
mock_main.vault = MagicMock()
mock_main.jenkins = MagicMock()
mock_main.utils = MagicMock()
sys.modules['main'] = mock_main
sys.modules['main.vault'] = mock_main.vault
sys.modules['main.jenkins'] = mock_main.jenkins
sys.modules['main.utils'] = mock_main.utils

# Import the function under test
from lambda_function import main


class TestLambdaAllInOne:
    """Complete test suite for the Lambda function using pytest-mock."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_event = {}
        
    def _setup_context(self, mocker):
        """Setup mock context using pytest-mock."""
        mock_context = mocker.MagicMock()
        mock_context.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test-function"
        return mock_context

    # === POSITIVE TESTS ===

    def test_successful_execution(self, mocker):
        """Test successful execution path."""
        # Setup mocks
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test', 'secret_key': 'test'}
        mock_get_jenkins_token.return_value = 'test-token'
        mock_call_jenkins.return_value = {
            'statusCode': 200,
            'body': json.dumps({'status': 'success'})
        }
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 200
        mock_get_ca_cert.assert_called_once_with('/tmp/ca-chain.cer')
        mock_get_aws_creds.assert_called_once_with('123456789012')
        mock_get_jenkins_token.assert_called_once_with('123456789012')
        mock_call_jenkins.assert_called_once()

    def test_successful_execution_complex_credentials(self, mocker):
        """Test successful execution with complex credentials."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        complex_aws_creds = {
            'access_key': 'AKIAIOSFODNN7EXAMPLE',
            'secret_key': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            'session_token': 'token123',
            'region': 'us-east-1'
        }
        mock_get_aws_creds.return_value = complex_aws_creds
        mock_get_jenkins_token.return_value = 'jenkins-token-12345'
        mock_call_jenkins.return_value = {
            'statusCode': 200,
            'body': json.dumps({'verification': 'passed'})
        }
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 200
        mock_call_jenkins.assert_called_once_with(
            aws_creds=complex_aws_creds,
            jenkins_token='jenkins-token-12345'
        )

    # === CERTIFICATE RETRIEVAL NEGATIVE TESTS ===
    
    def test_certificate_io_error(self, mocker):
        """Test IOError during certificate retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_ca_cert.side_effect = IOError("File write error")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 503
        assert 'System configuration error' in result['body']

    def test_certificate_os_error(self, mocker):
        """Test OSError during certificate retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_ca_cert.side_effect = OSError("Permission denied")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 503
        assert 'System configuration error' in result['body']

    def test_certificate_general_exception(self, mocker):
        """Test general exception during certificate retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_ca_cert.side_effect = Exception("Unexpected error")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 502
        assert 'Certificate configuration error' in result['body']

    # === CREDENTIAL RETRIEVAL NEGATIVE TESTS ===

    def test_connection_error(self, mocker):
        """Test ConnectionError during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = ConnectionError("Connection failed")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 503
        assert 'Authentication service unavailable' in result['body']

    def test_socket_gaierror(self, mocker):
        """Test socket.gaierror during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = socket.gaierror("Name resolution failed")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 503
        assert 'Authentication service unavailable' in result['body']

    def test_socket_timeout(self, mocker):
        """Test socket.timeout during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = socket.timeout()
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 504
        assert 'Authentication service timeout' in result['body']

    def test_value_error(self, mocker):
        """Test ValueError during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = ValueError("Invalid format")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 400
        assert 'Invalid credentials format' in result['body']

    def test_type_error_credentials(self, mocker):
        """Test TypeError during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = TypeError("Type mismatch")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 400
        assert 'Invalid credentials format' in result['body']

    def test_key_error(self, mocker):
        """Test KeyError during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = KeyError("missing_key")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 500
        assert 'Missing credentials' in result['body']

    def test_permission_error(self, mocker):
        """Test PermissionError during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = PermissionError("Access denied")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 403
        assert 'Permission denied' in result['body']

    def test_file_not_found_error(self, mocker):
        """Test FileNotFoundError during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = FileNotFoundError("Config not found")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 404
        assert 'Configuration not found' in result['body']

    def test_credential_general_exception(self, mocker):
        """Test general exception during credential retrieval."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.side_effect = RuntimeError("Unexpected error")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 500
        assert 'Authentication configuration error' in result['body']

    # === JENKINS CALL NEGATIVE TESTS ===

    def test_ssl_error(self, mocker):
        """Test SSL error during Jenkins call."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        mock_call_jenkins.side_effect = ssl.SSLError("SSL verification failed")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 495
        assert 'SSL verification failed' in result['body']

    def test_certificate_error_jenkins(self, mocker):
        """Test certificate error during Jenkins call - NOTE: CertificateError is caught by SSLError handler."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        mock_call_jenkins.side_effect = ssl.CertificateError("Certificate invalid")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 495
        # Note: CertificateError is a subclass of SSLError, so it gets caught by SSLError handler first
        assert 'SSL verification failed' in result['body']

    def test_url_error_connection_refused(self, mocker):
        """Test URLError with ConnectionRefusedError during Jenkins call."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        url_error = urllib_error.URLError(ConnectionRefusedError("Connection refused"))
        mock_call_jenkins.side_effect = url_error
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 503
        assert 'Jenkins service unavailable' in result['body']

    def test_url_error_timeout(self, mocker):
        """Test URLError with TimeoutError during Jenkins call."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        url_error = urllib_error.URLError(TimeoutError("Request timed out"))
        mock_call_jenkins.side_effect = url_error
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 504
        assert 'Jenkins request timed out' in result['body']

    def test_url_error_general(self, mocker):
        """Test URLError with general error during Jenkins call."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        url_error = urllib_error.URLError("General URL error")
        mock_call_jenkins.side_effect = url_error
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 502
        assert 'Jenkins service unreachable' in result['body']

    def test_http_error(self, mocker):
        """Test HTTPError during Jenkins call - NOTE: HTTPError is caught by URLError handler."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        http_error = urllib_error.HTTPError(
            url="http://test.com", 
            code=404, 
            msg="Not Found", 
            hdrs={}, 
            fp=None
        )
        mock_call_jenkins.side_effect = http_error
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        # Note: HTTPError is a subclass of URLError, so it gets caught by URLError handler first
        assert result['statusCode'] == 502
        assert 'Jenkins service unreachable' in result['body']

    def test_http_exception_jenkins(self, mocker):
        """Test HTTPException during Jenkins call."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        mock_call_jenkins.side_effect = http_client.HTTPException("Invalid HTTP response")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 502
        assert 'Invalid response from Jenkins' in result['body']

    def test_jenkins_general_exception(self, mocker):
        """Test general exception during Jenkins call."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        mock_call_jenkins.side_effect = Exception("Unexpected Jenkins error")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 500
        assert 'Internal server error' in result['body']

    def test_multiple_http_error_codes(self, mocker):
        """Test that HTTPError gets caught by URLError handler (inheritance behavior)."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        
        # Test that any HTTPError gets caught by URLError handler due to inheritance
        http_error = urllib_error.HTTPError(
            url="http://test.com", 
            code=500, 
            msg="Internal Server Error", 
            hdrs={}, 
            fp=None
        )
        mock_call_jenkins.side_effect = http_error
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        # HTTPError is caught by URLError handler first due to inheritance
        assert result['statusCode'] == 502
        assert 'Jenkins service unreachable' in result['body']

    # === EDGE CASE AND INTEGRATION TESTS ===

    def test_invalid_arn_format(self, mocker):
        """Test with invalid ARN format."""
        invalid_context = mocker.MagicMock()
        invalid_context.invoked_function_arn = "invalid-arn"
        
        # This should raise an IndexError when trying to split ARN
        with pytest.raises(IndexError):
            main(self.mock_event, invalid_context)

    def test_jenkins_token_failure_after_aws_creds_success(self, mocker):
        """Test failure when Jenkins token retrieval fails after AWS creds succeed."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.side_effect = ValueError("Invalid token format")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 400
        assert 'Invalid credentials format' in result['body']

    def test_certificate_success_then_credential_failure(self, mocker):
        """Test scenario where certificate succeeds but credentials fail."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        # Certificate succeeds
        mock_get_ca_cert.return_value = None
        
        # But credentials fail with timeout
        mock_get_aws_creds.side_effect = socket.timeout()
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 504
        assert 'Authentication service timeout' in result['body']
        # Verify certificate was called
        mock_get_ca_cert.assert_called_once_with('/tmp/ca-chain.cer')
        # But Jenkins was never called due to credential failure
        mock_call_jenkins.assert_not_called()

    def test_all_success_until_jenkins_failure(self, mocker):
        """Test scenario where everything succeeds until Jenkins call fails."""
        mock_get_ca_cert = mocker.patch('lambda_function.utils.get_ca_cert')
        mock_get_aws_creds = mocker.patch('lambda_function.vault.get_aws_creds')
        mock_get_jenkins_token = mocker.patch('lambda_function.vault.get_jenkins_token')
        mock_call_jenkins = mocker.patch('lambda_function.jenkins.call_jenkins_verify')
        
        # Everything succeeds initially
        mock_get_ca_cert.return_value = None
        mock_get_aws_creds.return_value = {'access_key': 'test'}
        mock_get_jenkins_token.return_value = 'token'
        
        # But Jenkins fails
        mock_call_jenkins.side_effect = ssl.SSLError("SSL handshake failed")
        
        mock_context = self._setup_context(mocker)
        result = main(self.mock_event, mock_context)
        
        assert result['statusCode'] == 495
        assert 'SSL verification failed' in result['body']
        
        # Verify all previous steps were called
        mock_get_ca_cert.assert_called_once()
        mock_get_aws_creds.assert_called_once()
        mock_get_jenkins_token.assert_called_once()
        mock_call_jenkins.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 