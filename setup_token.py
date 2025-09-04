"""
setup_token.py - Run this ONCE to get your refresh token
After running this, you'll never need to manually update tokens again!
"""

import os
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv, set_key

# Load existing .env
load_dotenv()

def setup_google_token():
    """One-time setup to get refresh token"""
    
    print("=" * 60)
    print("GOOGLE CALENDAR REFRESH TOKEN SETUP")
    print("=" * 60)
    
    # Get credentials from .env
    client_id = os.getenv('GOOGLE_CLIENT_ID')
    client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        print("\n❌ ERROR: Missing credentials in .env file")
        print("\nYou need to add to .env:")
        print("GOOGLE_CLIENT_ID=your_client_id")
        print("GOOGLE_CLIENT_SECRET=your_client_secret")
        print("\nGet these from Google Cloud Console:")
        print("1. Go to https://console.cloud.google.com")
        print("2. Create new project or select existing")
        print("3. Enable Google Calendar API")
        print("4. Create OAuth 2.0 Client ID (Desktop application)")
        print("5. Copy Client ID and Client Secret")
        return
    
    print(f"\n✅ Found Client ID: {client_id[:20]}...")
    print(f"✅ Found Client Secret: {client_secret[:10]}...")
    
    # Create OAuth flow
    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"]
            }
        },
        scopes=['https://www.googleapis.com/auth/calendar']
    )
    
    print("\n📌 A browser window will open for authorization...")
    print("Please sign in and grant calendar access.\n")
    
    try:
        # Run the OAuth flow
        credentials = flow.run_local_server(port=0)
        
        # Save tokens to .env
        print("\n✅ Authorization successful!")
        print("=" * 60)
        
        # Update .env file
        set_key('.env', 'GOOGLE_REFRESH_TOKEN', credentials.refresh_token)
        set_key('.env', 'GOOGLE_ACCESS_TOKEN', credentials.token)
        
        print("\n📝 Saved to .env file:")
        print(f"GOOGLE_REFRESH_TOKEN={credentials.refresh_token}")
        print(f"GOOGLE_ACCESS_TOKEN={credentials.token[:50]}...")
        
        # Save credentials to pickle file
        import pickle
        with open('google_token.pickle', 'wb') as token_file:
            pickle.dump(credentials, token_file)
        
        print("\n✅ Setup complete! Your bot will now auto-refresh tokens.")
        print("\n🚀 You can now run your bot with: python main.py")
        print("\n⚠️  IMPORTANT: Your refresh token is permanent unless revoked.")
        print("The bot will automatically get new access tokens when needed.")
        
    except Exception as e:
        print(f"\n❌ Error during authorization: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have Calendar API enabled")
        print("2. Check your Client ID and Secret are correct")
        print("3. Try using 'Desktop' type OAuth client")

if __name__ == "__main__":
    setup_google_token()