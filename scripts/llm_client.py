#!/usr/bin/env python3
"""
LLM Client Module

Provides a unified interface for interacting with either Azure OpenAI or OpenAI APIs.
Handles configuration, client creation, and abstracts provider-specific details.

IMPORTANT: Do not use max_tokens or temperature parameters when making API calls.
These parameters should never be specified - let the model use its defaults.

Usage:
    from llm_client import get_tag_generation_client, get_tag_assignment_client, get_title_generation_client

    client = get_tag_generation_client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "..."}],
        response_format={"type": "json_object"}
    )
"""

import os
import sys
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class LLMConfigError(Exception):
    """Raised when LLM configuration is invalid or incomplete."""
    pass


def _get_provider() -> str:
    """
    Get the LLM provider from environment variables.

    Returns:
        str: Either 'azure' or 'openai'

    Raises:
        LLMConfigError: If LLM_PROVIDER is not set or invalid
    """
    provider = os.getenv('LLM_PROVIDER', '').lower()

    if not provider:
        raise LLMConfigError(
            "LLM_PROVIDER environment variable is not set. "
            "Must be either 'azure' or 'openai'. "
            "Please check your .env file."
        )

    if provider not in ['azure', 'openai']:
        raise LLMConfigError(
            f"Invalid LLM_PROVIDER: '{provider}'. "
            "Must be either 'azure' or 'openai'."
        )

    return provider


def _validate_azure_config():
    """
    Validate that all required Azure OpenAI environment variables are set.

    Raises:
        LLMConfigError: If any required variable is missing
    """
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_KEY',
        'AZURE_OPENAI_API_VERSION',
        'AZURE_OPENAI_TAG_GENERATION_DEPLOYMENT',
        'AZURE_OPENAI_TAG_ASSIGNMENT_DEPLOYMENT',
        'AZURE_OPENAI_TITLE_GENERATION_DEPLOYMENT'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise LLMConfigError(
            f"Missing required Azure OpenAI environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all Azure variables are set."
        )


def _validate_openai_config():
    """
    Validate that all required OpenAI environment variables are set.

    Raises:
        LLMConfigError: If any required variable is missing
    """
    required_vars = [
        'OPENAI_API_KEY',
        'OPENAI_TAG_GENERATION_MODEL',
        'OPENAI_TAG_ASSIGNMENT_MODEL',
        'OPENAI_TITLE_GENERATION_MODEL'
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        raise LLMConfigError(
            f"Missing required OpenAI environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all OpenAI variables are set."
        )


def _create_azure_client(deployment_name: str):
    """
    Create an Azure OpenAI client configured for a specific deployment.

    Args:
        deployment_name: The Azure deployment name to use

    Returns:
        AzureOpenAI client instance

    Raises:
        LLMConfigError: If configuration is invalid
    """
    try:
        from openai import AzureOpenAI
    except ImportError:
        raise LLMConfigError(
            "The 'openai' package is not installed. "
            "Install it with: pip install openai"
        )

    _validate_azure_config()

    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_KEY')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION')

    # Create client with deployment name embedded
    # Note: Azure uses the model parameter to specify the deployment name
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    # Store deployment name for use in chat completions
    client._deployment_name = deployment_name

    # Wrap the chat.completions.create method to inject deployment name
    original_create = client.chat.completions.create

    def create_with_deployment(**kwargs):
        # Use 'model' parameter for deployment name in Azure
        if 'model' not in kwargs:
            kwargs['model'] = deployment_name
        return original_create(**kwargs)

    client.chat.completions.create = create_with_deployment

    return client


def _create_openai_client(model_name: str):
    """
    Create an OpenAI client configured for a specific model.

    Args:
        model_name: The OpenAI model name to use (e.g., 'gpt-4o', 'gpt-4o-mini')

    Returns:
        OpenAI client instance

    Raises:
        LLMConfigError: If configuration is invalid
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigError(
            "The 'openai' package is not installed. "
            "Install it with: pip install openai"
        )

    _validate_openai_config()

    api_key = os.getenv('OPENAI_API_KEY')

    client = OpenAI(api_key=api_key)

    # Store model name for use in chat completions
    client._model_name = model_name

    # Wrap the chat.completions.create method to inject model name
    original_create = client.chat.completions.create

    def create_with_model(**kwargs):
        if 'model' not in kwargs:
            kwargs['model'] = model_name
        return original_create(**kwargs)

    client.chat.completions.create = create_with_model

    return client


def get_tag_generation_client():
    """
    Get a client configured for tag generation.

    Returns:
        OpenAI or AzureOpenAI client instance

    Raises:
        LLMConfigError: If configuration is invalid

    Example:
        client = get_tag_generation_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Generate tags..."}],
            response_format={"type": "json_object"}
        )
    """
    provider = _get_provider()

    if provider == 'azure':
        deployment = os.getenv('AZURE_OPENAI_TAG_GENERATION_DEPLOYMENT')
        return _create_azure_client(deployment)
    else:  # openai
        model = os.getenv('OPENAI_TAG_GENERATION_MODEL')
        return _create_openai_client(model)


def get_tag_assignment_client():
    """
    Get a client configured for tag assignment.

    Returns:
        OpenAI or AzureOpenAI client instance

    Raises:
        LLMConfigError: If configuration is invalid

    Example:
        client = get_tag_assignment_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Assign tags..."}],
            response_format={"type": "json_object"}
        )
    """
    provider = _get_provider()

    if provider == 'azure':
        deployment = os.getenv('AZURE_OPENAI_TAG_ASSIGNMENT_DEPLOYMENT')
        return _create_azure_client(deployment)
    else:  # openai
        model = os.getenv('OPENAI_TAG_ASSIGNMENT_MODEL')
        return _create_openai_client(model)


def get_title_generation_client():
    """
    Get a client configured for title generation.

    Returns:
        OpenAI or AzureOpenAI client instance

    Raises:
        LLMConfigError: If configuration is invalid

    Example:
        client = get_title_generation_client()
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Generate title..."}],
            response_format={"type": "json_object"}
        )
    """
    provider = _get_provider()

    if provider == 'azure':
        deployment = os.getenv('AZURE_OPENAI_TITLE_GENERATION_DEPLOYMENT')
        return _create_azure_client(deployment)
    else:  # openai
        model = os.getenv('OPENAI_TITLE_GENERATION_MODEL')
        return _create_openai_client(model)


def get_provider_info() -> dict:
    """
    Get information about the current LLM provider configuration.

    Returns:
        dict: Provider information including type and model/deployment names

    Example:
        info = get_provider_info()
        print(f"Using {info['provider']} for LLM calls")
        print(f"Tag generation: {info['tag_generation']}")
    """
    provider = _get_provider()

    if provider == 'azure':
        return {
            'provider': 'azure',
            'endpoint': os.getenv('AZURE_OPENAI_ENDPOINT'),
            'api_version': os.getenv('AZURE_OPENAI_API_VERSION'),
            'tag_generation': os.getenv('AZURE_OPENAI_TAG_GENERATION_DEPLOYMENT'),
            'tag_assignment': os.getenv('AZURE_OPENAI_TAG_ASSIGNMENT_DEPLOYMENT'),
            'title_generation': os.getenv('AZURE_OPENAI_TITLE_GENERATION_DEPLOYMENT')
        }
    else:  # openai
        return {
            'provider': 'openai',
            'api_version': os.getenv('OPENAI_API_VERSION', 'latest'),
            'tag_generation': os.getenv('OPENAI_TAG_GENERATION_MODEL'),
            'tag_assignment': os.getenv('OPENAI_TAG_ASSIGNMENT_MODEL'),
            'title_generation': os.getenv('OPENAI_TITLE_GENERATION_MODEL')
        }


def _test_client(client, name: str, model_or_deployment: str) -> bool:
    """
    Test a client by making a simple API call.

    Args:
        client: The OpenAI or AzureOpenAI client to test
        name: Descriptive name for this client (e.g., "Tag Generation")
        model_or_deployment: The model or deployment name being tested

    Returns:
        bool: True if test succeeded, False otherwise
    """
    try:
        print(f"  Testing {name} ({model_or_deployment})...", end=" ", flush=True)

        # Make a minimal API call to test connectivity
        # NOTE: We intentionally do not use max_tokens or temperature parameters
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'test' and nothing else."}
            ]
        )

        # Verify we got a response
        if response.choices and len(response.choices) > 0:
            print("✓ Success")
            return True
        else:
            print("✗ Failed: No response received")
            return False

    except Exception as e:
        print(f"✗ Failed: {str(e)}")
        return False


if __name__ == '__main__':
    """Test the configuration and display provider info."""
    try:
        info = get_provider_info()
        print("=" * 80)
        print("LLM CONFIGURATION TEST")
        print("=" * 80)
        print()
        print("Configuration:")
        print(f"  Provider: {info['provider']}")

        if info['provider'] == 'azure':
            print(f"  Endpoint: {info['endpoint']}")
            print(f"  API Version: {info['api_version']}")
            print(f"  Tag Generation Deployment: {info['tag_generation']}")
            print(f"  Tag Assignment Deployment: {info['tag_assignment']}")
            print(f"  Title Generation Deployment: {info['title_generation']}")
        else:
            print(f"  API Version: {info['api_version']}")
            print(f"  Tag Generation Model: {info['tag_generation']}")
            print(f"  Tag Assignment Model: {info['tag_assignment']}")
            print(f"  Title Generation Model: {info['title_generation']}")

        print()
        print("Testing API connectivity...")
        print()

        # Track which models/deployments we've already tested
        tested = set()
        all_passed = True

        # Test tag generation
        tag_gen_name = info['tag_generation']
        if tag_gen_name not in tested:
            client = get_tag_generation_client()
            if not _test_client(client, "Tag Generation", tag_gen_name):
                all_passed = False
            tested.add(tag_gen_name)
        else:
            print(f"  Tag Generation ({tag_gen_name})... ✓ Already tested")

        # Test tag assignment
        tag_assign_name = info['tag_assignment']
        if tag_assign_name not in tested:
            client = get_tag_assignment_client()
            if not _test_client(client, "Tag Assignment", tag_assign_name):
                all_passed = False
            tested.add(tag_assign_name)
        else:
            print(f"  Tag Assignment ({tag_assign_name})... ✓ Already tested")

        # Test title generation
        title_gen_name = info['title_generation']
        if title_gen_name not in tested:
            client = get_title_generation_client()
            if not _test_client(client, "Title Generation", title_gen_name):
                all_passed = False
            tested.add(title_gen_name)
        else:
            print(f"  Title Generation ({title_gen_name})... ✓ Already tested")

        print()
        print("=" * 80)
        if all_passed:
            print("✓ All tests passed! Configuration is valid and API is accessible.")
        else:
            print("✗ Some tests failed. Please check your configuration and API access.")
            sys.exit(1)
        print("=" * 80)

    except LLMConfigError as e:
        print()
        print("=" * 80)
        print(f"✗ Configuration Error: {e}", file=sys.stderr)
        print("=" * 80)
        sys.exit(1)
