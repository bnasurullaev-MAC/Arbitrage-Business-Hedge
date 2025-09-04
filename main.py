"""
Enhanced Calendar Bot with Conversation Context Management
Maintains conversation history for better understanding of follow-up questions
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
import re
import aiohttp
import calendar
import pytz
import pickle
from collections import defaultdict, deque

from dotenv import load_dotenv, set_key
load_dotenv()

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
import google.generativeai as genai

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalendarStates(StatesGroup):
    confirming_event = State()
    selecting_event = State()
    processing_multiple_actions = State()

class ConversationManager:
    """Manages conversation history for each user"""
    
    def __init__(self, max_history_length: int = 10, max_tokens: int = 2000):
        self.conversations = defaultdict(lambda: deque(maxlen=max_history_length))
        self.max_tokens = max_tokens
        self.conversations_file = 'conversation_history.json'
        self.load_conversations()
    
    def load_conversations(self):
        """Load saved conversations from file"""
        try:
            if os.path.exists(self.conversations_file):
                with open(self.conversations_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, messages in data.items():
                        self.conversations[int(user_id)] = deque(messages, maxlen=10)
                logger.info(f"Loaded conversations for {len(self.conversations)} users")
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
    
    def save_conversations(self):
        """Save conversations to file"""
        try:
            data = {str(user_id): list(messages) for user_id, messages in self.conversations.items()}
            with open(self.conversations_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def add_message(self, user_id: int, role: str, content: str):
        """Add a message to user's conversation history"""
        self.conversations[user_id].append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        self.save_conversations()
    
    def get_context(self, user_id: int, num_messages: int = 6) -> str:
        """Get formatted conversation context for the user"""
        messages = list(self.conversations[user_id])[-num_messages:]
        
        if not messages:
            return ""
        
        context = "Previous conversation:\n"
        for msg in messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            # Truncate very long messages
            content = msg['content']
            if len(content) > 500:
                content = content[:500] + "..."
            context += f"{role}: {content}\n"
        
        return context
    
    def clear_history(self, user_id: int):
        """Clear conversation history for a user"""
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            self.save_conversations()
    
    def get_last_topic(self, user_id: int) -> Optional[str]:
        """Extract the likely topic from recent conversation"""
        messages = list(self.conversations[user_id])[-3:]
        if messages:
            # Look for calendar-related keywords or general topics
            recent_text = " ".join([m['content'] for m in messages])
            return recent_text[-200:]  # Last 200 chars for context
        return None

class GoogleTokenManager:
    """Manages Google OAuth tokens with automatic refresh"""
    
    def __init__(self):
        self.token_file = 'google_token.pickle'
        self.credentials = None
        self.client_id = os.getenv('GOOGLE_CLIENT_ID')
        self.client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
        self.refresh_token = os.getenv('GOOGLE_REFRESH_TOKEN')
        self.load_credentials()
    
    def load_credentials(self):
        """Load saved credentials from pickle file"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'rb') as token:
                    self.credentials = pickle.load(token)
                logger.info("Loaded saved Google credentials")
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
                self.create_credentials_from_env()
        else:
            self.create_credentials_from_env()
    
    def create_credentials_from_env(self):
        """Create credentials from environment variables"""
        if self.refresh_token and self.client_id and self.client_secret:
            self.credentials = Credentials(
                token=os.getenv('GOOGLE_ACCESS_TOKEN'),
                refresh_token=self.refresh_token,
                token_uri='https://oauth2.googleapis.com/token',
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            self.save_credentials()
    
    def save_credentials(self):
        """Save credentials to pickle file"""
        try:
            with open(self.token_file, 'wb') as token:
                pickle.dump(self.credentials, token)
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
    
    def get_valid_token(self) -> str:
        """Get a valid access token, refreshing if necessary"""
        if not self.credentials:
            return None
        
        if self.credentials.expired or not self.credentials.token:
            try:
                self.credentials.refresh(Request())
                self.save_credentials()
                set_key('.env', 'GOOGLE_ACCESS_TOKEN', self.credentials.token)
                logger.info("Token refreshed successfully")
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                return None
        
        return self.credentials.token

class ImprovedCalendarAgent:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.token_manager = GoogleTokenManager()
        
        self.calendar_api_base = "https://www.googleapis.com/calendar/v3"
        self.timezone = pytz.timezone('America/Chicago')
        
        self.bot = Bot(token=self.bot_token)
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager()
        
        # Store event search results for better selection
        self.last_event_search = {}
        # Store pending multiple actions
        self.pending_multiple_actions = {}
        
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.gemini_model = None
        
        self.setup_handlers()
        logger.info("Improved Calendar Agent with Context Management initialized")
    
    def get_access_token(self) -> str:
        """Get valid access token"""
        return self.token_manager.get_valid_token()
    
    def load_json_file(self, filename: str, default_value):
        """Load JSON file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
        return default_value
    
    def save_json_file(self, filename: str, data):
        """Save JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
    
    def parse_duration(self, text: str) -> float:
        """Parse duration from text - returns hours as float"""
        text_lower = text.lower()
        
        patterns = [
            (r'(\d+\.?\d*)\s*hours?', 1),
            (r'(\d+\.?\d*)\s*hrs?', 1),
            (r'(\d+)\s*minutes?', 1/60),
            (r'(\d+)\s*mins?', 1/60),
            (r'half\s*(?:an\s*)?hour', 0.5),
        ]
        
        for pattern, multiplier in patterns:
            if isinstance(multiplier, (int, float)):
                match = re.search(pattern, text_lower)
                if match:
                    if pattern == r'half\s*(?:an\s*)?hour':
                        return 0.5
                    value = float(match.group(1))
                    return value * multiplier
        
        return 1.0
    
    async def process_with_gemini(self, user_message: str, user_id: int) -> str:
        """Enhanced Gemini processing with conversation context"""
        
        if not self.gemini_model:
            return "I need Gemini API to work properly."
        
        try:
            current_date = datetime.now(self.timezone).strftime('%A, %B %d, %Y')
            current_time = datetime.now(self.timezone).strftime('%I:%M %p %Z')
            
            # Get conversation context
            context = self.conversation_manager.get_context(user_id)
            
            # Build the prompt with context
            prompt = f"""You are an AI assistant that helps with calendar management and general questions.
Today is {current_date} at {current_time}.

{context}

Current user message: "{user_message}"

IMPORTANT INSTRUCTIONS:
1. Consider the conversation history when responding. If the user asks a follow-up question (like "what about X"), understand it in context of the previous messages.
2. Maintain conversation continuity - recognize when users are continuing a previous topic.
3. For general Q&A or chat, respond naturally without calendar actions unless specifically requested.
4. Only create calendar actions when the user explicitly asks for calendar operations.

For CALENDAR operations only, use these action tags:
- CREATE: [CALENDAR_ACTION: CREATE_EVENT | title: "title" | date: "date" | time: "time" | duration: "duration"]
- VIEW: [CALENDAR_ACTION: VIEW_EVENTS | range: "today/tomorrow/week/month"]
- DELETE: [CALENDAR_ACTION: DELETE_EVENT | description: "specific event description with time"]
- UPDATE: [CALENDAR_ACTION: UPDATE_EVENT | old_description: "current event" | new_date: "date" | new_time: "time" | new_duration: "duration"]

Examples of context-aware responses:

User: "Does Claude AI have limits even with paid subscriptions?"
Assistant: "Yes, even with paid subscriptions, Claude AI has usage limits..."
User: "What about ChatGPT?"
Assistant: [Understanding this is about ChatGPT's limits] "Yes, ChatGPT also has limits even with paid subscriptions..."

User: "I need help with Q&A"
Assistant: "I can help with Q&A! What questions do you have?"
User: "Does Claude have limits?"
Assistant: [Understanding this is part of the Q&A] "Yes, Claude has usage limits..."

Remember:
- If the user message is ambiguous, use the conversation history to understand what they're referring to
- Maintain the topic flow from previous messages
- Don't ask what they're talking about if it's clear from context

Respond to the user naturally:"""

            response = await self.gemini_model.generate_content_async(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"I encountered an error processing your message. Please try again."
    
    def parse_calendar_actions(self, response: str) -> Tuple[str, List[Dict]]:
        """Extract multiple calendar actions from response"""
        
        action_pattern = r'\[CALENDAR_ACTION:\s*([^\]]+)\]'
        action_matches = re.findall(action_pattern, response)
        
        if not action_matches:
            return response, []
        
        clean_response = re.sub(action_pattern, '', response).strip()
        
        actions = []
        for action_text in action_matches:
            parts = action_text.split('|')
            action_type = parts[0].strip()
            details = {}
            
            for part in parts[1:]:
                if ':' in part:
                    key, value = part.split(':', 1)
                    details[key.strip()] = value.strip().strip('"')
            
            actions.append({'type': action_type, 'details': details})
        
        return clean_response, actions
    
    def parse_date(self, date_str: str) -> datetime:
        """Parse date from string"""
        date_lower = date_str.lower()
        now = datetime.now(self.timezone)
        
        if 'today' in date_lower:
            return now
        elif 'tomorrow' in date_lower:
            return now + timedelta(days=1)
        
        days = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6
        }
        
        for day_name, day_num in days.items():
            if day_name in date_lower:
                days_ahead = day_num - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                return now + timedelta(days=days_ahead)
        
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'sept': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        for month_name, month_num in months.items():
            if month_name in date_lower:
                day_match = re.search(r'(\d{1,2})', date_str)
                if day_match:
                    day = int(day_match.group(1))
                    year = now.year
                    try:
                        result = self.timezone.localize(datetime(year, month_num, day))
                        if result.date() < now.date():
                            result = self.timezone.localize(datetime(year + 1, month_num, day))
                        return result
                    except:
                        pass
        
        return now
    
    def parse_time(self, time_str: str) -> Tuple[int, int]:
        """Parse time string and return hour, minute"""
        time_lower = time_str.lower()
        
        hour, minute = 14, 0
        
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_lower)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2) or 0)
            period = time_match.group(3)
            
            if period == 'pm' and hour != 12:
                hour += 12
            elif period == 'am' and hour == 12:
                hour = 0
            elif not period and hour < 8:
                hour += 12
        
        return hour, minute
    
    async def handle_calendar_actions(self, actions: List[Dict], message: types.Message, state: FSMContext):
        """Execute multiple calendar actions with better state management"""
        
        create_actions = [a for a in actions if a['type'] == 'CREATE_EVENT']
        delete_actions = [a for a in actions if a['type'] == 'DELETE_EVENT']
        update_actions = [a for a in actions if a['type'] == 'UPDATE_EVENT']
        view_actions = [a for a in actions if a['type'] == 'VIEW_EVENTS']
        
        for action in view_actions:
            await self.handle_view_events(action['details'].get('range', 'today'), message)
        
        if create_actions:
            await self.handle_multiple_creates(create_actions, message, state)
        elif delete_actions:
            await self.handle_multiple_deletes(delete_actions, message)
        elif update_actions:
            await self.handle_multiple_updates(update_actions, message, state)
    
    async def handle_multiple_creates(self, create_actions: List[Dict], message: types.Message, state: FSMContext):
        """Handle multiple event creations"""
        events_to_create = []
        
        for action in create_actions:
            details = action['details']
            duration_str = details.get('duration', '1 hour')
            duration_hours = self.parse_duration(duration_str)
            
            event_data = {
                'title': details.get('title', 'Event'),
                'date': self.parse_date(details.get('date', 'today')),
                'time': details.get('time', '2pm'),
                'duration_hours': duration_hours,
                'duration_str': duration_str
            }
            events_to_create.append(event_data)
        
        confirm_msg = f"📅 **Confirm Creation of {len(events_to_create)} Event(s):**\n\n"
        
        for i, event in enumerate(events_to_create, 1):
            date_str = event['date'].strftime('%A, %B %d, %Y')
            if event['duration_hours'] < 1:
                duration_display = f"{int(event['duration_hours'] * 60)} minutes"
            else:
                duration_display = f"{event['duration_hours']} hour{'s' if event['duration_hours'] != 1 else ''}"
            
            confirm_msg += f"**{i}. {event['title']}**\n"
            confirm_msg += f"   Date: {date_str}\n"
            confirm_msg += f"   Time: {event['time']}\n"
            confirm_msg += f"   Duration: {duration_display}\n\n"
        
        confirm_msg += "Reply 'yes' to create all or 'no' to cancel."
        
        await state.update_data(pending_events=events_to_create)
        await state.set_state(CalendarStates.confirming_event)
        await message.answer(confirm_msg)
    
    async def handle_multiple_deletes(self, delete_actions: List[Dict], message: types.Message):
        """Handle multiple event deletions"""
        delete_msg = "🗑 **Processing deletion requests...**\n\n"
        
        for action in delete_actions:
            description = action['details'].get('description', '')
            events = await self.find_events_for_deletion(description)
            
            if events:
                for event in events:
                    if await self.delete_event(event['id']):
                        delete_msg += f"✅ Deleted: {event.get('summary', 'Event')}\n"
                    else:
                        delete_msg += f"❌ Failed to delete: {event.get('summary', 'Event')}\n"
            else:
                delete_msg += f"❌ No events found matching: {description}\n"
        
        await message.answer(delete_msg)
    
    async def handle_multiple_updates(self, update_actions: List[Dict], message: types.Message, state: FSMContext):
        """Handle multiple event updates with better conflict resolution"""
        
        all_updates = []
        ambiguous_updates = []
        
        for action in update_actions:
            details = action['details']
            old_desc = details.get('old_description', '')
            events = await self.find_events_for_deletion(old_desc)
            
            if not events:
                await message.answer(f"❌ Couldn't find event: {old_desc}")
            elif len(events) == 1:
                all_updates.append({
                    'event': events[0],
                    'details': details,
                    'action': action
                })
            else:
                ambiguous_updates.append({
                    'events': events[:5],
                    'details': details,
                    'action': action,
                    'description': old_desc
                })
        
        if all_updates:
            update_msg = "📝 **Updating events...**\n\n"
            for update_item in all_updates:
                new_duration = None
                if 'new_duration' in update_item['details']:
                    new_duration = self.parse_duration(update_item['details']['new_duration'])
                
                result = await self.update_event(
                    update_item['event']['id'],
                    new_title=update_item['details'].get('new_title'),
                    new_date_str=update_item['details'].get('new_date'),
                    new_time=update_item['details'].get('new_time'),
                    new_duration=new_duration
                )
                
                if result['success']:
                    update_msg += f"✅ {result.get('message', 'Updated successfully')}\n"
                else:
                    update_msg += f"❌ Failed: {result.get('error')}\n"
            
            await message.answer(update_msg)
        
        if ambiguous_updates:
            user_id = message.from_user.id
            self.pending_multiple_actions[user_id] = {
                'ambiguous_updates': ambiguous_updates,
                'current_index': 0
            }
            
            await self.show_next_ambiguous_update(message, state)
    
    async def show_next_ambiguous_update(self, message: types.Message, state: FSMContext):
        """Show the next ambiguous update that needs user selection"""
        user_id = message.from_user.id
        
        if user_id not in self.pending_multiple_actions:
            return
        
        pending = self.pending_multiple_actions[user_id]
        ambiguous_updates = pending.get('ambiguous_updates', [])
        current_index = pending.get('current_index', 0)
        
        if current_index >= len(ambiguous_updates):
            del self.pending_multiple_actions[user_id]
            await state.clear()
            await message.answer("✅ All updates completed!")
            return
        
        current_update = ambiguous_updates[current_index]
        events = current_update['events']
        
        response = f"🔍 Found multiple events matching '{current_update['description']}'\n"
        response += f"Select which one to update (or type 'skip' to skip this update, 'cancel' to stop):\n\n"
        
        for i, event in enumerate(events, 1):
            title = event.get('summary', 'Untitled')
            start = event.get('start', {})
            if 'dateTime' in start:
                dt = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
                dt_local = dt.astimezone(self.timezone)
                time_str = dt_local.strftime('%b %d at %I:%M %p')
            else:
                time_str = "All day"
            response += f"{i}. {title} - {time_str}\n"
        
        self.last_event_search[user_id] = events
        await state.set_state(CalendarStates.processing_multiple_actions)
        await state.update_data(
            action='update_multiple',
            update_details=current_update['details'],
            update_index=current_index
        )
        
        await message.answer(response)
    
    async def handle_view_events(self, range_filter: str, message: types.Message):
        """View events with better formatting"""
        events = await self.get_calendar_events(range_filter)
        
        if events['success']:
            if not events['events']:
                await message.answer(f"No events found for {range_filter}.")
            else:
                response = f"📅 Your {range_filter}'s events:\n\n"
                for i, event in enumerate(events['events'][:15], 1):
                    title = event.get('summary', 'Untitled')
                    start = event.get('start', {})
                    
                    if 'dateTime' in start:
                        dt_str = start['dateTime']
                        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        dt_local = dt.astimezone(self.timezone)
                        
                        if range_filter == 'week':
                            time_str = dt_local.strftime('%a %b %d, %I:%M %p')
                        else:
                            time_str = dt_local.strftime('%I:%M %p')
                    else:
                        time_str = "All day"
                    
                    response += f"{i}. {time_str} - {title}\n"
                
                await message.answer(response)
        else:
            await message.answer("Failed to fetch events.")
    
    async def find_events_for_deletion(self, description: str) -> List[Dict]:
        """Better event search for deletion/update"""
        
        date_found = None
        time_found = None
        
        if any(word in description.lower() for word in ['today', 'tomorrow', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']):
            date_found = self.parse_date(description)
        
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        for month in months:
            if month in description.lower():
                date_found = self.parse_date(description)
                break
        
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', description.lower())
        if time_match:
            time_found = time_match.group(0)
        
        if date_found:
            result = await self.get_calendar_events_for_date(date_found)
        else:
            result = await self.get_calendar_events('week')
        
        if not result['success'] or not result['events']:
            return []
        
        matching_events = []
        description_lower = description.lower()
        
        for event in result['events']:
            score = 0
            event_title = event.get('summary', '').lower()
            
            title_words = event_title.split()
            desc_words = description_lower.split()
            for word in title_words:
                if word in desc_words:
                    score += 2
            
            if time_found and 'dateTime' in event.get('start', {}):
                dt = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
                dt_local = dt.astimezone(self.timezone)
                event_time = dt_local.strftime('%I:%M%p').lower().replace(' ', '').replace(':00', '')
                if time_found.replace(' ', '').replace(':00', '') in event_time:
                    score += 5
            
            if date_found and 'dateTime' in event.get('start', {}):
                dt = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
                dt_local = dt.astimezone(self.timezone)
                if dt_local.date() == date_found.date():
                    score += 3
            
            if score > 0:
                matching_events.append((event, score))
        
        matching_events.sort(key=lambda x: x[1], reverse=True)
        return [event for event, score in matching_events[:5]]
    
    async def get_calendar_events_for_date(self, date: datetime) -> Dict:
        """Get events for specific date"""
        
        token = self.get_access_token()
        if not token:
            return {'success': False, 'error': 'No valid token'}
        
        try:
            start = date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
            
            start_utc = start.astimezone(pytz.UTC)
            end_utc = end.astimezone(pytz.UTC)
            
            params = {
                'timeMin': start_utc.isoformat().replace('+00:00', 'Z'),
                'timeMax': end_utc.isoformat().replace('+00:00', 'Z'),
                'singleEvents': 'true',
                'orderBy': 'startTime',
                'maxResults': '50'
            }
            
            url = f"{self.calendar_api_base}/calendars/primary/events?" + '&'.join([f'{k}={v}' for k, v in params.items()])
            
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'success': True, 'events': data.get('items', [])}
                    else:
                        return {'success': False, 'error': f'Failed: {response.status}'}
                        
        except Exception as e:
            logger.error(f"Get events error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_calendar_events(self, range_filter: str) -> Dict:
        """Get calendar events for a range"""
        
        token = self.get_access_token()
        if not token:
            return {'success': False, 'error': 'No valid token'}
        
        try:
            now = datetime.now(self.timezone)
            
            if 'week' in range_filter:
                start = now - timedelta(days=now.weekday())
                start = start.replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=7)
            elif 'tomorrow' in range_filter:
                start = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1)
            elif 'month' in range_filter:
                start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if now.month == 12:
                    end = start.replace(year=start.year + 1, month=1)
                else:
                    end = start.replace(month=start.month + 1)
            else:
                start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1)
            
            start_utc = start.astimezone(pytz.UTC)
            end_utc = end.astimezone(pytz.UTC)
            
            params = {
                'timeMin': start_utc.isoformat().replace('+00:00', 'Z'),
                'timeMax': end_utc.isoformat().replace('+00:00', 'Z'),
                'singleEvents': 'true',
                'orderBy': 'startTime',
                'maxResults': '50'
            }
            
            url = f"{self.calendar_api_base}/calendars/primary/events?" + '&'.join([f'{k}={v}' for k, v in params.items()])
            
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {'success': True, 'events': data.get('items', [])}
                    else:
                        return {'success': False, 'error': f'Failed: {response.status}'}
                        
        except Exception as e:
            logger.error(f"Get events error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def delete_event(self, event_id: str) -> bool:
        """Delete calendar event"""
        
        token = self.get_access_token()
        if not token:
            return False
        
        try:
            url = f"{self.calendar_api_base}/calendars/primary/events/{event_id}"
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=headers) as response:
                    return response.status == 204
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    async def update_event(self, event_id: str, new_title: str = None,
                          new_date_str: str = None, new_time: str = None,
                          new_duration: float = None) -> Dict:
        """Update calendar event with duration support"""
        
        token = self.get_access_token()
        if not token:
            return {'success': False, 'error': 'No valid token'}
        
        try:
            url = f"{self.calendar_api_base}/calendars/primary/events/{event_id}"
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return {'success': False, 'error': 'Event not found'}
                    
                    current_event = await response.json()
                
                updated_event = current_event.copy()
                
                if new_title:
                    updated_event['summary'] = new_title
                
                if new_date_str or new_time or new_duration is not None:
                    current_start = current_event.get('start', {})
                    if 'dateTime' in current_start:
                        current_dt = datetime.fromisoformat(current_start['dateTime'].replace('Z', '+00:00'))
                        current_dt_local = current_dt.astimezone(self.timezone)
                    else:
                        current_dt_local = datetime.now(self.timezone)
                    
                    if 'dateTime' in current_event.get('end', {}):
                        current_end = datetime.fromisoformat(current_event['end']['dateTime'].replace('Z', '+00:00'))
                        current_start_for_duration = datetime.fromisoformat(current_event['start']['dateTime'].replace('Z', '+00:00'))
                        current_duration = (current_end - current_start_for_duration).total_seconds() / 3600
                    else:
                        current_duration = 1
                    
                    if new_date_str:
                        new_date = self.parse_date(new_date_str)
                        hour = current_dt_local.hour
                        minute = current_dt_local.minute
                    else:
                        new_date = current_dt_local
                        hour = current_dt_local.hour
                        minute = current_dt_local.minute
                    
                    if new_time:
                        hour, minute = self.parse_time(new_time)
                    
                    duration_hours = new_duration if new_duration is not None else current_duration
                    
                    new_start_dt = new_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if not new_start_dt.tzinfo:
                        new_start_dt = self.timezone.localize(new_start_dt)
                    
                    new_end_dt = new_start_dt + timedelta(hours=duration_hours)
                    
                    new_start_utc = new_start_dt.astimezone(pytz.UTC)
                    new_end_utc = new_end_dt.astimezone(pytz.UTC)
                    
                    updated_event['start'] = {
                        'dateTime': new_start_utc.isoformat().replace('+00:00', 'Z'),
                        'timeZone': str(self.timezone)
                    }
                    updated_event['end'] = {
                        'dateTime': new_end_utc.isoformat().replace('+00:00', 'Z'),
                        'timeZone': str(self.timezone)
                    }
                
                async with session.put(url, headers=headers, json=updated_event) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        title = result.get('summary', 'Event')
                        start = result.get('start', {})
                        if 'dateTime' in start:
                            dt = datetime.fromisoformat(start['dateTime'].replace('Z', '+00:00'))
                            dt_local = dt.astimezone(self.timezone)
                            time_str = dt_local.strftime('%B %d at %I:%M %p')
                        else:
                            time_str = "Date updated"
                        
                        return {
                            'success': True,
                            'message': f"Updated '{title}' to {time_str}"
                        }
                    else:
                        return {'success': False, 'error': f'Failed: {response.status}'}
                        
        except Exception as e:
            logger.error(f"Update error: {e}")
            return {'success': False, 'error': str(e)}
    
    async def create_event(self, event_data: Dict) -> Dict:
        """Create calendar event with proper duration"""
        
        token = self.get_access_token()
        if not token:
            return {'success': False, 'error': 'No valid token'}
        
        try:
            hour, minute = self.parse_time(event_data['time'])
            
            duration_hours = event_data.get('duration_hours', 1)
            
            start_dt = event_data['date'].replace(hour=hour, minute=minute, second=0, microsecond=0)
            if not start_dt.tzinfo:
                start_dt = self.timezone.localize(start_dt)
            
            end_dt = start_dt + timedelta(hours=duration_hours)
            
            start_utc = start_dt.astimezone(pytz.UTC)
            end_utc = end_dt.astimezone(pytz.UTC)
            
            event = {
                'summary': event_data['title'],
                'start': {
                    'dateTime': start_utc.isoformat().replace('+00:00', 'Z'),
                    'timeZone': str(self.timezone)
                },
                'end': {
                    'dateTime': end_utc.isoformat().replace('+00:00', 'Z'),
                    'timeZone': str(self.timezone)
                },
                'description': f'Created by AI Assistant - Duration: {event_data.get("duration_str", "1 hour")}'
            }
            
            url = f"{self.calendar_api_base}/calendars/primary/events"
            headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
            
            logger.info(f"Creating event: {event}")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=event) as response:
                    if response.status in [200, 201]:
                        result = await response.json()
                        return {'success': True, 'event_id': result['id']}
                    else:
                        text = await response.text()
                        logger.error(f"Failed to create: {text}")
                        return {'success': False, 'error': f'Failed: {response.status}'}
                        
        except Exception as e:
            logger.error(f"Create error: {e}")
            return {'success': False, 'error': str(e)}
    
    def setup_handlers(self):
        """Setup message handlers"""
        
        @self.dp.message(Command("start"))
        async def start_command(message: types.Message):
            welcome = f"""👋 Hi! I'm your AI assistant with calendar capabilities.

I can help with:
📅 **Calendar Management:**
• Create, update, delete events
• View your schedule
• Handle multiple events at once

💬 **General Chat & Q&A:**
• Answer questions on various topics
• Have natural conversations
• Remember our conversation context

Try asking me anything or managing your calendar!"""
            
            # Clear conversation history on start
            self.conversation_manager.clear_history(message.from_user.id)
            await message.answer(welcome)
        
        @self.dp.message(Command("clear"))
        async def clear_command(message: types.Message):
            """Clear conversation history"""
            self.conversation_manager.clear_history(message.from_user.id)
            await message.answer("✅ Conversation history cleared. Starting fresh!")
        
        @self.dp.message(Command("history"))
        async def history_command(message: types.Message):
            """Show conversation history"""
            context = self.conversation_manager.get_context(message.from_user.id, num_messages=10)
            if context:
                await message.answer(f"📜 **Recent conversation:**\n\n{context[-1500:]}")  # Limit to last 1500 chars
            else:
                await message.answer("No conversation history yet.")
        
        @self.dp.message(CalendarStates.confirming_event)
        async def confirm_event(message: types.Message, state: FSMContext):
            """Handle event confirmation for single or multiple events"""
            text = message.text.lower()
            
            if text in ['yes', 'y', 'confirm', 'ok', 'sure']:
                data = await state.get_data()
                
                pending_events = data.get('pending_events')
                if pending_events:
                    results_msg = "📅 **Creating events...**\n\n"
                    success_count = 0
                    fail_count = 0
                    
                    for event_data in pending_events:
                        result = await self.create_event(event_data)
                        
                        if result['success']:
                            success_count += 1
                            date_str = event_data['date'].strftime('%B %d')
                            results_msg += f"✅ Created: {event_data['title']} on {date_str}\n"
                        else:
                            fail_count += 1
                            results_msg += f"❌ Failed: {event_data['title']} - {result.get('error')}\n"
                    
                    results_msg += f"\n**Summary:** {success_count} created, {fail_count} failed"
                    await message.answer(results_msg)
                else:
                    event_data = data.get('pending_event')
                    if event_data:
                        result = await self.create_event(event_data)
                        
                        if result['success']:
                            await message.answer(f"✅ Event created!\nID: {result['event_id'][:20]}...")
                        else:
                            await message.answer(f"❌ Failed: {result.get('error')}")
                
                await state.clear()
                
            elif text in ['no', 'n', 'cancel']:
                await message.answer("Cancelled.")
                await state.clear()
            else:
                await message.answer("Please reply 'yes' or 'no'")
        
        @self.dp.message(CalendarStates.processing_multiple_actions)
        async def process_multiple_action_selection(message: types.Message, state: FSMContext):
            """Handle selection during multiple action processing"""
            text = message.text.strip().lower()
            user_id = message.from_user.id
            
            if text == 'cancel':
                if user_id in self.pending_multiple_actions:
                    del self.pending_multiple_actions[user_id]
                if user_id in self.last_event_search:
                    del self.last_event_search[user_id]
                await state.clear()
                await message.answer("❌ All remaining updates cancelled.")
                return
            
            if text == 'skip':
                if user_id in self.pending_multiple_actions:
                    pending = self.pending_multiple_actions[user_id]
                    pending['current_index'] += 1
                    await self.show_next_ambiguous_update(message, state)
                return
            
            try:
                choice = int(text)
                
                if user_id in self.last_event_search and 1 <= choice <= len(self.last_event_search[user_id]):
                    selected_event = self.last_event_search[user_id][choice - 1]
                    data = await state.get_data()
                    
                    update_details = data.get('update_details', {})
                    new_duration = None
                    if 'new_duration' in update_details:
                        new_duration = self.parse_duration(update_details['new_duration'])
                    
                    result = await self.update_event(
                        selected_event['id'],
                        new_title=update_details.get('new_title'),
                        new_date_str=update_details.get('new_date'),
                        new_time=update_details.get('new_time'),
                        new_duration=new_duration
                    )
                    
                    if result['success']:
                        await message.answer(f"✅ {result.get('message')}")
                    else:
                        await message.answer(f"❌ Failed: {result.get('error')}")
                    
                    if user_id in self.pending_multiple_actions:
                        pending = self.pending_multiple_actions[user_id]
                        pending['current_index'] += 1
                        await self.show_next_ambiguous_update(message, state)
                else:
                    await message.answer("Invalid choice. Enter a number, 'skip', or 'cancel'.")
                    
            except ValueError:
                await message.answer("Please enter a number, 'skip', or 'cancel'.")
        
        @self.dp.message(CalendarStates.selecting_event)
        async def select_event(message: types.Message, state: FSMContext):
            """Handle single event selection for delete/update"""
            text = message.text.strip().lower()
            
            if text == 'cancel':
                user_id = message.from_user.id
                if user_id in self.last_event_search:
                    del self.last_event_search[user_id]
                await state.clear()
                await message.answer("Cancelled.")
                return
            
            try:
                choice = int(text)
                user_id = message.from_user.id
                
                if user_id in self.last_event_search and 1 <= choice <= len(self.last_event_search[user_id]):
                    selected_event = self.last_event_search[user_id][choice - 1]
                    data = await state.get_data()
                    action = data.get('action')
                    
                    if action == 'delete':
                        if await self.delete_event(selected_event['id']):
                            await message.answer(f"✅ Deleted: {selected_event.get('summary', 'Event')}")
                        else:
                            await message.answer("Failed to delete.")
                    elif action == 'update':
                        update_details = data.get('update_details', {})
                        new_duration = None
                        if 'new_duration' in update_details:
                            new_duration = self.parse_duration(update_details['new_duration'])
                        
                        result = await self.update_event(
                            selected_event['id'],
                            new_title=update_details.get('new_title'),
                            new_date_str=update_details.get('new_date'),
                            new_time=update_details.get('new_time'),
                            new_duration=new_duration
                        )
                        
                        if result['success']:
                            await message.answer(f"✅ {result.get('message')}")
                        else:
                            await message.answer(f"❌ Failed: {result.get('error')}")
                    
                    del self.last_event_search[user_id]
                    await state.clear()
                else:
                    await message.answer("Invalid choice. Please enter a number from the list or 'cancel'.")
                    
            except ValueError:
                await message.answer("Please enter a number from the list or 'cancel'.")
            except Exception as e:
                logger.error(f"Selection error: {e}")
                await message.answer("Error processing selection.")
                await state.clear()
        
        @self.dp.message(F.text)
        async def handle_any_message(message: types.Message, state: FSMContext):
            """Handle all messages with conversation context"""
            
            user_id = message.from_user.id
            user_message = message.text
            
            # Add user message to conversation history
            self.conversation_manager.add_message(user_id, 'user', user_message)
            
            await self.bot.send_chat_action(message.chat.id, 'typing')
            
            # Process with Gemini including context
            ai_response = await self.process_with_gemini(user_message, user_id)
            
            # Parse calendar actions if any
            clean_response, calendar_actions = self.parse_calendar_actions(ai_response)
            
            # Add assistant response to conversation history
            self.conversation_manager.add_message(user_id, 'assistant', clean_response)
            
            if clean_response:
                await message.answer(clean_response)
            
            # Handle calendar actions if any
            if calendar_actions:
                await self.handle_calendar_actions(calendar_actions, message, state)
    
    async def run(self):
        """Start the bot"""
        print("🚀 Starting Enhanced Calendar Bot with Conversation Context...")
        print("✨ Features: Context-aware conversations + Calendar management")
        try:
            await self.dp.start_polling(self.bot)
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            await self.bot.session.close()

def main():
    """Main function"""
    print("=" * 60)
    print("🤖 AI Assistant with Calendar & Conversation Context")
    print("=" * 60)
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    gemini_key = os.getenv('GEMINI_API_KEY')
    
    print(f"Bot Token: {'✅' if bot_token else '❌'}")
    print(f"Gemini AI: {'✅' if gemini_key else '❌'}")
    print(f"Google Auth: {'✅' if os.path.exists('google_token.pickle') else '⚠️'}")
    
    if not bot_token or not gemini_key:
        print("\n❌ Required tokens missing!")
        return
    
    print("\n✨ Key features:")
    print("• Conversation context management")
    print("• Natural follow-up question understanding")
    print("• General Q&A capabilities")
    print("• Calendar event management")
    print("• Multiple event handling\n")
    
    print("Commands:")
    print("• /start - Initialize bot")
    print("• /clear - Clear conversation history")
    print("• /history - Show recent conversation\n")
    
    async def start():
        agent = ImprovedCalendarAgent()
        await agent.run()
    
    try:
        asyncio.run(start())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped")

if __name__ == '__main__':
    main()