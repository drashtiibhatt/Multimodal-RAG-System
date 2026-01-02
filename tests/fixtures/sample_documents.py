"""
Sample documents for testing.

Provides realistic test data for document parsing, chunking, and retrieval tests.
"""

from src.ingestion.parsers.base import Document


# ==================== Authentication Domain ====================

AUTH_SPEC_DOCUMENT = Document(
    page_content="""
User Authentication System Specification

Overview:
The authentication system provides secure user login and session management
functionality for the application.

Requirements:

1. Login Functionality
   - Users must provide email and password
   - Email must be in valid format (RFC 5322)
   - Password must meet complexity requirements:
     * Minimum 8 characters
     * At least one uppercase letter
     * At least one lowercase letter
     * At least one number
     * At least one special character

2. Session Management
   - Sessions are managed using JWT tokens
   - Token expiry: 24 hours
   - Refresh token expiry: 7 days
   - Tokens must be stored securely (httpOnly cookies)

3. Security Features
   - Passwords hashed using bcrypt (cost factor: 12)
   - Account lockout after 5 failed login attempts
   - Lockout duration: 15 minutes
   - HTTPS required for all authentication endpoints
   - Rate limiting: 10 requests per minute per IP

4. Two-Factor Authentication (Optional)
   - SMS-based OTP (6 digits, 5-minute expiry)
   - Authenticator app support (TOTP)
   - Backup codes (10 single-use codes)

Error Handling:
- Invalid credentials: "Invalid email or password"
- Account locked: "Account temporarily locked. Try again in X minutes"
- Expired session: "Session expired. Please log in again"
- Invalid OTP: "Invalid verification code"

API Endpoints:
- POST /auth/login - Authenticate user
- POST /auth/logout - Terminate session
- POST /auth/refresh - Refresh access token
- POST /auth/verify-2fa - Verify OTP for 2FA
""",
    metadata={
        "source": "auth_specification.md",
        "file_type": "text",
        "domain": "authentication",
        "version": "1.0"
    }
)


REGISTRATION_SPEC_DOCUMENT = Document(
    page_content="""
User Registration Specification

Process Flow:

1. User Provides Information
   - Email address (required)
   - Password (required, must meet strength requirements)
   - Username (required, 3-20 alphanumeric characters)
   - Full name (optional)
   - Phone number (optional, for 2FA)

2. Validation
   - Check email format validity
   - Verify email not already registered
   - Validate password strength
   - Verify username availability
   - Check username for profanity/reserved words

3. Account Creation
   - Hash password using bcrypt
   - Generate unique user ID (UUID v4)
   - Create database record with status: 'pending_verification'
   - Generate email verification token (expires in 24 hours)

4. Email Verification
   - Send verification email with unique link
   - Link format: /verify-email?token=<verification_token>
   - Token must be single-use
   - If not verified within 24 hours, account deleted

5. Post-Verification
   - Update account status to 'active'
   - Send welcome email
   - Create default user preferences
   - Log registration event

Error Scenarios:
- Email already registered: "Email already in use"
- Username taken: "Username not available"
- Weak password: "Password does not meet requirements"
- Invalid email: "Invalid email address"
- Verification link expired: "Link expired. Please register again"

Test Data:
Valid email: test@example.com
Valid password: SecurePass123!
Valid username: testuser2024
""",
    metadata={
        "source": "registration_specification.md",
        "file_type": "text",
        "domain": "registration",
        "version": "1.0"
    }
)


PASSWORD_RESET_DOCUMENT = Document(
    page_content="""
Password Reset Flow

Initiate Reset:
1. User clicks "Forgot Password" link
2. User enters registered email address
3. System validates email exists in database
4. Generate password reset token (secure random, 32 bytes)
5. Token expires in 1 hour
6. Send password reset email with link
7. Link format: /reset-password?token=<reset_token>

Security Measures:
- Generic message shown regardless of email existence
  (prevents email enumeration)
- Rate limit: 3 reset requests per hour per IP
- Tokens are single-use only
- Token invalidated after successful reset
- Previous sessions terminated after reset

Reset Password:
1. User clicks link in email
2. Validate token (not expired, not used)
3. User enters new password (must be different from old)
4. Validate password strength
5. Hash new password with bcrypt
6. Update password in database
7. Invalidate reset token
8. Terminate all active sessions
9. Send confirmation email
10. Redirect to login page

Error Cases:
- Invalid token: "Invalid or expired reset link"
- Token expired: "Reset link expired. Please request a new one"
- Token already used: "Reset link already used"
- Same password: "New password must be different from old password"
- Weak password: "Password does not meet requirements"
""",
    metadata={
        "source": "password_reset.md",
        "file_type": "text",
        "domain": "password_management",
        "version": "1.0"
    }
)


# ==================== E-commerce Domain ====================

PRODUCT_CATALOG_DOCUMENT = Document(
    page_content="""
Product Catalog Management

Product Attributes:
- Product ID (unique, auto-generated)
- SKU (Stock Keeping Unit, unique)
- Name (max 200 characters)
- Description (max 5000 characters)
- Category (hierarchical, up to 3 levels)
- Price (decimal, 2 decimal places)
- Discount percentage (0-100)
- Stock quantity (integer, >= 0)
- Images (up to 10 images, max 5MB each)
- Variants (size, color, etc.)
- Status (draft, active, out_of_stock, discontinued)

Search and Filtering:
- Full-text search on name and description
- Filter by category, price range, availability
- Sort by price, popularity, newest, rating
- Pagination (20 items per page)

Inventory Management:
- Automatic stock updates on order
- Low stock alerts (threshold: 10 units)
- Out of stock handling (backorder or hide)
- Stock reservation during checkout (15-minute hold)

Product Lifecycle:
1. Draft - being prepared by admin
2. Active - available for purchase
3. Out of Stock - temporarily unavailable
4. Discontinued - no longer sold
""",
    metadata={
        "source": "product_catalog.md",
        "file_type": "text",
        "domain": "ecommerce",
        "version": "1.0"
    }
)


# ==================== Payment Domain ====================

PAYMENT_PROCESSING_DOCUMENT = Document(
    page_content="""
Payment Processing Specification

Supported Payment Methods:
1. Credit/Debit Cards (Visa, MasterCard, Amex)
2. Digital Wallets (PayPal, Apple Pay, Google Pay)
3. Bank Transfer (ACH, Wire)
4. Buy Now Pay Later (Klarna, Afterpay)

Payment Flow:
1. User selects items and proceeds to checkout
2. Calculate total: subtotal + tax + shipping - discounts
3. User selects payment method
4. Collect payment details
5. Validate payment information
6. Process payment via payment gateway
7. Handle response (success, failure, pending)
8. Update order status
9. Send confirmation email
10. Trigger fulfillment

Card Payment Validation:
- Card number: Luhn algorithm check
- Expiry date: Must be in future
- CVV: 3-4 digits
- Billing address: Must match card
- 3D Secure for transactions > $100

Security:
- PCI DSS Level 1 compliance
- Tokenization of card details
- Encryption in transit (TLS 1.3)
- Fraud detection (velocity checks, geolocation)
- No storage of CVV

Error Handling:
- Insufficient funds: Prompt to try different card
- Card declined: "Payment declined. Please contact your bank"
- Invalid card: "Invalid card information"
- Network error: Retry mechanism (3 attempts)
- Timeout: "Payment processing timeout. Please try again"

Refund Process:
- Full refund: within 30 days
- Partial refund: for damaged items
- Processing time: 5-7 business days
- Automatic notification to customer
""",
    metadata={
        "source": "payment_processing.md",
        "file_type": "text",
        "domain": "payment",
        "version": "1.0"
    }
)


# ==================== Sample Document Lists ====================

ALL_AUTH_DOCUMENTS = [
    AUTH_SPEC_DOCUMENT,
    REGISTRATION_SPEC_DOCUMENT,
    PASSWORD_RESET_DOCUMENT
]

ALL_ECOMMERCE_DOCUMENTS = [
    PRODUCT_CATALOG_DOCUMENT,
    PAYMENT_PROCESSING_DOCUMENT
]

ALL_SAMPLE_DOCUMENTS = ALL_AUTH_DOCUMENTS + ALL_ECOMMERCE_DOCUMENTS


# ==================== Helper Functions ====================

def get_documents_by_domain(domain: str):
    """Get all documents for a specific domain."""
    return [doc for doc in ALL_SAMPLE_DOCUMENTS
            if doc.metadata.get('domain') == domain]


def get_document_by_source(source: str):
    """Get document by source filename."""
    for doc in ALL_SAMPLE_DOCUMENTS:
        if doc.metadata.get('source') == source:
            return doc
    return None
