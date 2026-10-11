import datetime
import random
from typing import List

from langchain_community.tools import tool

@tool
def check_employee_account(employee_id: str) -> str:
    """Check the account status of an employee"""
    return f"Employee {employee_id} is active"

@tool
def check_vpn_status(employee_id: str) -> str:
    """Check the VPN status for an employee"""
    return f"Employee {employee_id} has access to the VPN"

@tool
def create_ticket(issue_description: str) -> str:
    """Create a support ticket for an issue"""
    return f"Support ticket created for issue: {issue_description}"

@tool
def reset_password(employee_id: str) -> str:
    """Reset the password for an employee"""
    return f"Password reset for employee {employee_id}"
    
@tool
def generate_sample_users(
        first_names: List[str],
        last_names: List[str],
        domains: List[str],
        min_age: int,
        max_age: int) -> dict:
    """
    Generate sample user data. Count is determined by the length of first_names.

    Args:
        first_names: List of first names (one per user)
        last_names: List of last names (will cycle if fewer than first_names)
        domains: List of email domains (will cycle through)
        min_age: Minimum age for users
        max_age: Maximum age for users

    Returns:
        Dictionary with 'users' array or 'error' message
    """
    # Validation
    if not first_names:
        return {"error": "first_names list cannot be empty"}
    if not last_names:
        return {"error": "last_names list cannot be empty"}
    if not domains:
        return {"error": "domains list cannot be empty"}
    if min_age > max_age:
        return {"error": f"min_age ({min_age}) cannot be greater than max_age ({max_age})"}
    if min_age < 0 or max_age < 0:
        return {"error": "ages must be non-negative"}

    users = []
    count = len(first_names)

    for i in range(count):
        first = first_names[i]
        last = last_names[i % len(last_names)]
        domain = domains[i % len(domains)]
        email = f"{first.lower()}.{last.lower()}@{domain}"

        user = {
            "id": i + 1,
            "firstName": first,
            "lastName": last,
            "email": email,
            "username": f"{first.lower()}{random.randint(100, 999)}",
            "age": random.randint(min_age, max_age),
            "registeredAt": (datetime.now() - datetime.timedelta(days=random.randint(1, 365))).isoformat()
        }
        users.append(user)

    return {"users": users, "count": len(users)}

ALL_TOOLS = [check_employee_account, check_vpn_status, create_ticket, reset_password]
print('Mode with Tools are ready')