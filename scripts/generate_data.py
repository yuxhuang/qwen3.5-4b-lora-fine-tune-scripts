#!/usr/bin/env python3
"""
Generate diverse, realistic ReAct-format training data for the agentic loop:
fetch_url -> save_page -> extract_dom_content -> validate_content.

Uses modular page builder functions with content pools for combinatorial diversity.
Each call to a builder produces a unique (url, html, selector, expected_text) tuple.

Default: 2500 examples (~2060 train / ~440 eval), ~22% failure cases across 5 modes.
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_JSON = PROJECT_ROOT / "tools" / "definitions.json"
DATA_DIR = PROJECT_ROOT / "data"


def load_tool_definitions() -> dict:
    with open(TOOLS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def format_tools_for_prompt(tools_def: dict) -> str:
    lines = ["Available tools (JSON Schema):"]
    for t in tools_def["tools"]:
        lines.append(f"- {t['name']}: {t['description']}")
        lines.append(f"  Parameters: {json.dumps(t['parameters'], indent=2)}")
    return "\n".join(lines)


# ============================================================================
# Content pools for combinatorial diversity
# ============================================================================

COMPANY_NAMES = [
    "ShopMart", "TechNova", "CloudPeak", "DataForge", "NexGen Labs",
    "BlueWave Systems", "IronPulse", "ClearPath Analytics", "Summit Tools",
    "Velocity Corp", "PrimeLogic", "ArcLight", "ZenithOS", "Orion Digital",
    "Meridian Tech", "Apex Software", "BrightEdge", "QuantumLeap", "SilverLine",
    "RedPanda AI", "Atlas Commerce", "Forge Industries", "Helix Solutions",
]

PERSON_NAMES = [
    "Sarah Chen", "James Liu", "Maria Garcia", "Robert Williams", "Emily Zhang",
    "David Kim", "Anna Petrov", "Michael O'Brien", "Priya Sharma", "Carlos Mendez",
    "Lisa Johnson", "Tom Baker", "Yuki Tanaka", "Ahmed Hassan", "Rachel Green",
    "Nikolai Volkov", "Sophie Martin", "Jake Thompson", "Mei Wong", "Alex Rivera",
]

PRODUCT_NAMES = [
    "ProSound Wireless Headphones", "SpeedRunner Pro Shoes", "ErgoLift Laptop Stand",
    "UltraView 4K Monitor", "NightOwl Desk Lamp", "AeroFit Smart Watch",
    "CloudSync External SSD", "TrueColor Stylus Pen", "PowerCore Battery Pack",
    "SilentType Mechanical Keyboard", "ClearTone Speaker System", "SwiftCharge Adapter",
    "ArcticBreeze Portable Fan", "PixelPro Graphics Tablet", "FlexGrip Phone Mount",
    "EchoBeam Projector", "VoltEdge Gaming Mouse", "ThermoSeal Travel Mug",
    "NanoClean Air Purifier", "LumiTrack Fitness Band", "QuietComfort Earbuds",
    "SmartBrew Coffee Maker", "FlexDock USB-C Hub", "TitanShield Laptop Case",
]

PRICES = [
    "$9.99", "$14.50", "$24.99", "$29.00", "$34.50", "$45.00", "$49.99",
    "$64.95", "$79.99", "$89.00", "$99.97", "$124.50", "$149.99", "$179.00",
    "$199.95", "$249.00", "$299.99", "$349.50", "$449.00", "$599.99", "$749.00",
]

RATINGS = ["3.8/5", "4.0/5", "4.2/5", "4.5/5", "4.7/5", "4.8/5", "4.9/5"]
REVIEW_COUNTS = ["42 reviews", "128 reviews", "238 reviews", "567 reviews", "1,204 reviews", "3,891 reviews"]
STOCK_STATUSES = ["In Stock", "In Stock - 3 left", "In Stock - Ships in 24h", "Low Stock - Only 2 remaining"]
OUT_OF_STOCK = ["Out of Stock", "Sold Out", "Currently Unavailable", "Backordered"]

ARTICLE_TITLES = [
    "AI Trends to Watch in 2026", "The Future of Edge Computing", "Quantum Computing Breakthrough",
    "How Remote Work Changed Tech Hiring", "Cybersecurity Threats Rising in Healthcare",
    "Open Source LLMs Reach New Milestone", "5G Rollout Hits Urban Areas",
    "Blockchain in Supply Chain Management", "The State of DevOps in 2026",
    "Machine Learning in Drug Discovery", "Climate Tech Startups Raise Record Funding",
    "Electric Vehicles Outsell Gas Cars in Europe", "Space Tourism Opens to Public",
    "Gene Therapy Approved for Rare Disease", "Ocean Cleanup Project Hits Milestone",
    "Renewable Energy Now Cheapest Option", "Digital Currency Adoption Accelerates",
    "Autonomous Vehicles Get Green Light in California", "Browser Wars Heat Up Again",
    "New Programming Language Gains Traction", "Wearable Health Monitors Save Lives",
]

DATES = [
    "January 5, 2026", "January 18, 2026", "February 2, 2026", "February 14, 2026",
    "February 28, 2026", "March 1, 2026", "March 5, 2026", "March 8, 2026",
    "March 10, 2026", "December 15, 2025", "November 20, 2025", "October 3, 2025",
]

STATUS_OPERATIONAL = ["Operational", "All Systems Normal", "Healthy", "Running"]
STATUS_DEGRADED = ["Degraded Performance", "Partial Outage", "Elevated Error Rates", "Slow Response Times"]
STATUS_DOWN = ["Major Outage", "Service Unavailable", "Down", "Critical"]

TICKET_IDS = ["TK-88712", "TK-44201", "TK-99130", "TK-12847", "TK-55603", "TK-77891", "TK-33456"]
TICKET_SUBJECTS = [
    "Cannot export CSV reports", "Login page returns 502", "Dashboard charts not loading",
    "Email notifications delayed by 2 hours", "API rate limiting too aggressive",
    "Search results returning stale data", "Mobile app crashes on iOS 19",
    "Webhook deliveries failing intermittently", "File upload stuck at 99%",
    "Two-factor authentication code not arriving",
]
PRIORITIES = ["Critical", "High", "Medium", "Low"]
TICKET_STATUSES = ["Open", "In Progress", "Waiting on Customer", "Under Review", "Escalated"]

CITIES = [
    "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", "Chicago, IL",
    "Boston, MA", "Denver, CO", "Portland, OR", "Miami, FL", "Atlanta, GA",
    "London, UK", "Berlin, Germany", "Tokyo, Japan", "Sydney, Australia", "Toronto, Canada",
]

TEMPERATURES = ["45F", "52F", "58F", "62F", "68F", "72F", "78F", "84F", "91F"]
CONDITIONS = [
    "Sunny", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Light Rain",
    "Heavy Rain", "Thunderstorms", "Snow", "Foggy", "Clear",
]

JOB_TITLES = [
    "Senior Software Engineer", "Staff Data Scientist", "Product Manager",
    "DevOps Lead", "Frontend Architect", "Security Engineer", "ML Engineer",
    "Engineering Manager", "Technical Program Manager", "Solutions Architect",
    "Platform Engineer", "Site Reliability Engineer", "UX Designer",
]

SALARY_RANGES = [
    "$120,000 - $160,000", "$140,000 - $190,000", "$160,000 - $220,000",
    "$180,000 - $250,000", "$200,000 - $280,000", "$220,000 - $300,000",
]

DEPARTMENTS = ["Engineering", "Data Science", "Product", "Design", "Security", "Platform", "Infrastructure"]

COURSE_CODES = ["CS101", "CS201", "CS301", "CS401", "DS200", "DS350", "ML400", "SE250"]
COURSE_NAMES = [
    "Introduction to Computer Science", "Data Structures and Algorithms",
    "Machine Learning Fundamentals", "Advanced Operating Systems",
    "Statistical Learning", "Deep Learning", "Natural Language Processing",
    "Software Engineering Practices", "Distributed Systems", "Computer Networks",
]

INSTRUCTORS = [f"Dr. {n}" for n in PERSON_NAMES[:12]]

FLIGHT_CODES = ["UA1234", "DL567", "AA890", "SW321", "JB456", "NK789", "AS234", "WN678"]
AIRPORTS = [
    ("SFO", "San Francisco"), ("JFK", "New York"), ("LAX", "Los Angeles"),
    ("ORD", "Chicago"), ("DFW", "Dallas"), ("SEA", "Seattle"),
    ("BOS", "Boston"), ("ATL", "Atlanta"), ("DEN", "Denver"), ("MIA", "Miami"),
]
FLIGHT_STATUSES = ["On Time", "Delayed 30 min", "Delayed 1 hour", "Boarding", "Departed", "Arrived"]
GATES = ["A12", "B24", "C7", "D15", "E3", "F22", "A8", "B31", "C19"]

BALANCE_AMOUNTS = [
    "$1,247.83", "$2,891.50", "$4,287.53", "$6,103.22", "$8,450.00",
    "$12,340.67", "$18,920.41", "$24,500.00", "$31,872.95", "$47,210.88",
]

ACCOUNT_TYPES = ["Checking Account", "Savings Account", "Money Market Account", "Investment Account"]

METRIC_VALUES_USERS = ["2,847", "5,120", "8,341", "12,847", "24,901", "45,230", "98,412"]
METRIC_VALUES_REVENUE = ["$84,392", "$142,500", "$284,392", "$521,000", "$892,341", "$1.2M", "$2.4M"]
CONVERSION_RATES = ["1.8%", "2.3%", "3.1%", "3.7%", "4.2%", "5.1%", "6.8%"]
SESSION_COUNTS = ["234", "567", "1,203", "2,841", "5,120", "8,901"]

DOMAINS = [
    "shopmart.com", "technova.io", "cloudpeak.dev", "dataforge.ai", "nexgenlabs.com",
    "bluewavesys.io", "ironpulse.dev", "clearpath.co", "summittools.com", "velocitycorp.io",
    "primelogic.dev", "arclight.io", "zenithos.com", "orion-digital.net", "meridiantech.co",
    "apexsoft.io", "brightedge.dev", "quantumleap.ai", "silverline.co", "redpanda.ai",
    "atlas-commerce.com", "forgeind.io", "helix-solutions.dev", "globalwidgets.com",
    "fastship.co", "healthtrack.io", "eduplatform.dev", "finserve.co", "travelgo.com",
]

PATH_PREFIXES = [
    "/data/pages", "/var/www/saved", "/tmp/fetched", "/home/user/pages",
    "/data/archive", "/opt/scraping/output", "/data/crawl", "/home/agent/html",
]

CUISINE_ITEMS = [
    ("Grilled Salmon", "$24.99"), ("Ribeye Steak", "$34.50"), ("Veggie Bowl", "$16.99"),
    ("Pad Thai", "$18.50"), ("Caesar Salad", "$12.99"), ("Margherita Pizza", "$15.00"),
    ("Chicken Tikka Masala", "$19.99"), ("Sushi Platter", "$28.00"), ("Lamb Chops", "$32.00"),
    ("Mushroom Risotto", "$21.50"), ("Tacos Al Pastor", "$14.99"), ("Tom Yum Soup", "$11.50"),
]

PODCAST_TITLES = [
    "The Rise of Edge Computing", "Debugging Production at Scale", "Open Source Business Models",
    "Building Developer Tools", "AI Ethics in Practice", "The Future of Databases",
    "Remote Team Management", "Security in the Cloud Era", "Data Engineering Patterns",
    "Startup Lessons Learned", "The API Economy", "Modern Frontend Architecture",
]

PLAY_COUNTS = ["1,204", "3,420", "7,891", "12,340", "24,500", "45,000", "89,210"]
DURATIONS = ["22 min", "35 min", "42 min", "45 min", "58 min", "1 hr 12 min"]

PLAYER_NAMES = [
    "xDragonSlayer", "NinjaCoderX", "StarPilot99", "PhoenixRise", "ShadowByte",
    "QuantumFox", "IronWolf42", "CyberEagle", "NeonTiger", "VortexKing",
]

SCORES = ["45,200", "67,815", "78,420", "89,100", "92,340", "96,200", "97,815", "98,420"]

EMAIL_SENDERS = [f"{n.split()[0]} {n.split()[1]}" for n in PERSON_NAMES]
EMAIL_SUBJECTS = [
    "Project update for Q1", "Meeting notes from standup", "Budget approval needed",
    "New hire onboarding", "Sprint retrospective action items", "Quarterly review prep",
    "Client feedback summary", "Infrastructure migration plan", "Security audit results",
    "Release v3.2 notes", "Vendor contract renewal", "Team offsite planning",
]

TRACKING_IDS = ["PKG-881234", "PKG-220198", "PKG-554312", "PKG-998871", "PKG-113456", "PKG-667823"]
DELIVERY_STATUSES = ["Out for Delivery", "In Transit", "Delivered", "Processing", "Shipped"]
DELIVERY_ETAS = ["Today by 5:00 PM", "Tomorrow by noon", "March 12, 2026", "March 15, 2026", "2-3 business days"]

MOVIE_TITLES = [
    "Interstellar 2: Beyond Time", "The Last Algorithm", "Neon City", "Echoes of Tomorrow",
    "The Deep Web", "Quantum Heist", "Atlas Rising", "Silent Protocol",
    "The Infinite Loop", "Darknet", "Binary Stars", "Code Zero",
]

MOVIE_SCORES = ["72%", "78%", "82%", "85%", "88%", "91%", "92%", "95%", "97%"]

LISTING_TYPES = ["3BR Home", "2BR Condo", "4BR House", "Studio Apartment", "Townhouse", "Loft"]
LISTING_PRICES = ["$425,000", "$550,000", "$675,000", "$750,000", "$875,000", "$1,200,000", "$1,850,000"]
NEIGHBORHOODS = [
    "Oakland Hills", "Capitol Hill", "Nob Hill", "Back Bay", "Wicker Park",
    "Silver Lake", "Ballard", "South End", "Buckhead", "Cherry Creek",
]

PLAN_NAMES = ["Free", "Starter", "Professional", "Business", "Enterprise"]
PLAN_PRICES = ["$0/mo", "$9/mo", "$29/mo", "$79/mo", "$149/mo", "Custom"]

HVAC_MODES = ["Cooling", "Heating", "Auto", "Fan Only", "Eco"]
TEMPS = ["65F", "68F", "70F", "72F", "74F", "76F"]

INCIDENT_TITLES = [
    "Database Latency Spike", "CDN Cache Invalidation Failure", "API Gateway Timeout",
    "Payment Processing Delay", "Search Index Corruption", "Email Delivery Queue Backup",
    "Auth Service Memory Leak", "Storage Cluster Failover", "DNS Resolution Failure",
    "Load Balancer Health Check Flapping",
]
SEVERITIES = ["Critical", "Major", "Minor", "Maintenance"]

FORUM_TOPICS = [
    "Best practices for Docker in production", "How to debug memory leaks in Node.js",
    "Migrating from MySQL to PostgreSQL", "Understanding Kubernetes networking",
    "Optimizing Python for data pipelines", "React vs Vue in 2026",
    "Troubleshooting CORS issues", "Setting up CI/CD with GitHub Actions",
    "gRPC vs REST for microservices", "MongoDB aggregation pipeline tips",
    "Terraform state management strategies", "Redis caching best practices",
]

REPLY_COUNTS = ["3 replies", "7 replies", "12 replies", "24 replies", "48 replies", "96 replies"]
VIEW_COUNTS = ["142 views", "520 views", "1,280 views", "3,450 views", "8,901 views", "15,200 views"]

CHANGELOG_VERSIONS = ["v3.2.0", "v3.1.4", "v3.1.3", "v3.0.0", "v2.9.8", "v2.8.0", "v4.0.0-beta"]
CHANGELOG_ITEMS = [
    "Added support for WebSocket connections",
    "Fixed memory leak in background worker",
    "Improved query performance by 40%",
    "Added dark mode support",
    "Fixed CSV export encoding issue",
    "Upgraded authentication to OAuth 2.1",
    "Added bulk import API endpoint",
    "Fixed race condition in job scheduler",
    "Improved error messages for validation failures",
    "Added support for custom webhooks",
    "Removed deprecated v1 API endpoints",
    "Fixed timezone handling in date filters",
]

SEARCH_QUERIES = [
    "machine learning tutorial", "kubernetes deployment yaml", "react hooks guide",
    "python async await", "docker compose networking", "rust error handling",
    "typescript generics", "go concurrency patterns", "sql window functions",
    "css grid layout", "api rate limiting", "jwt authentication",
]

SEARCH_RESULT_TITLES = [
    "Complete Guide to {q}", "Understanding {q} from Scratch",
    "{q} - Official Documentation", "A Practical Introduction to {q}",
    "{q}: Tips and Tricks", "Advanced {q} Techniques",
]

NOTIFICATION_TYPES = [
    ("mentioned you in", "a comment on PR #421"),
    ("approved your", "pull request #389"),
    ("assigned you to", "issue #1204"),
    ("commented on", "your merge request"),
    ("requested your review on", "PR #567"),
    ("closed", "issue #892 as completed"),
    ("merged", "branch feature/auth into main"),
    ("deployed", "v3.2.0 to production"),
]

EVENT_NAMES = [
    "Engineering All-Hands", "Sprint Planning Q2", "Design Review: Dashboard Redesign",
    "Security Incident Retro", "Product Roadmap Review", "Tech Talk: Scaling Databases",
    "1:1 with Manager", "Architecture Decision Record Review", "Release Readiness Meeting",
    "Customer Feedback Sync", "Budget Planning Session", "Hackathon Kickoff",
]

EVENT_TIMES = [
    "10:00 AM - 11:00 AM", "2:00 PM - 3:00 PM", "11:30 AM - 12:30 PM",
    "3:30 PM - 4:30 PM", "9:00 AM - 10:30 AM", "1:00 PM - 2:00 PM",
]

EVENT_LOCATIONS = [
    "Zoom (link in calendar)", "Room 4A - Main Office", "Google Meet",
    "Conference Room B", "Microsoft Teams", "Room 201 - Building 3",
]

CART_QUANTITIES = ["1", "2", "3", "4", "5"]
CART_TOTALS = ["$47.98", "$89.97", "$124.50", "$189.99", "$234.48", "$312.00", "$459.97"]
PROMO_CODES = ["SAVE10", "SPRING25", "WELCOME15", "FREESHIP", "VIP20"]
DISCOUNT_AMOUNTS = ["-$5.00", "-$10.00", "-$15.00", "-$25.00", "-$47.50"]

API_ENDPOINTS = [
    ("GET", "/api/v2/users", "List all users"),
    ("POST", "/api/v2/users", "Create a new user"),
    ("GET", "/api/v2/users/{id}", "Get user by ID"),
    ("PUT", "/api/v2/users/{id}", "Update user"),
    ("DELETE", "/api/v2/users/{id}", "Delete user"),
    ("GET", "/api/v2/orders", "List orders"),
    ("POST", "/api/v2/orders", "Create order"),
    ("GET", "/api/v2/products", "List products"),
    ("GET", "/api/v2/analytics/summary", "Get analytics summary"),
    ("POST", "/api/v2/webhooks", "Register webhook"),
]

RATE_LIMITS = ["100 req/min", "500 req/min", "1000 req/min", "5000 req/hr"]

RECIPE_NAMES = [
    "Classic Beef Bourguignon", "Thai Green Curry", "Homemade Sourdough Bread",
    "Mediterranean Quinoa Bowl", "Crispy Fried Chicken", "Vegetable Stir Fry",
    "Lemon Garlic Salmon", "Mushroom Risotto", "Chicken Tikka Masala",
    "Chocolate Lava Cake", "Eggs Benedict", "Pad Thai Noodles",
]

PREP_TIMES = ["15 min", "20 min", "25 min", "30 min", "45 min"]
COOK_TIMES = ["20 min", "30 min", "45 min", "1 hour", "1 hr 30 min", "2 hours"]
RECIPE_SERVINGS = ["2 servings", "4 servings", "6 servings", "8 servings"]
RECIPE_RATINGS = ["4.2/5 (89 ratings)", "4.5/5 (234 ratings)", "4.7/5 (567 ratings)", "4.9/5 (1,203 ratings)"]
DIFFICULTY_LEVELS = ["Easy", "Intermediate", "Advanced"]

WORKOUT_TYPES = [
    "Upper Body Strength", "HIIT Cardio Blast", "Core & Abs Circuit",
    "Full Body Burn", "Lower Body Power", "Yoga Flow", "Endurance Run",
    "Chest & Triceps", "Back & Biceps", "Leg Day",
]

WORKOUT_DURATIONS = ["20 min", "30 min", "45 min", "60 min", "75 min"]
CALORIES_BURNED = ["180 kcal", "250 kcal", "320 kcal", "420 kcal", "550 kcal", "680 kcal"]
HEART_RATES = ["128 bpm", "142 bpm", "155 bpm", "165 bpm", "172 bpm"]

WIKI_TITLES = [
    "Getting Started Guide", "Configuration Reference", "API Authentication",
    "Deployment Architecture", "Database Schema", "Error Code Reference",
    "Troubleshooting Common Issues", "Migration Guide v2 to v3",
    "Security Best Practices", "Performance Tuning", "Plugin Development",
    "Backup and Recovery", "Webhook Integration", "CLI Reference",
]

WIKI_UPDATED_BY = [f"Edited by {n}" for n in PERSON_NAMES[:10]]

SURVEY_QUESTIONS = [
    "How satisfied are you with our product?", "How likely are you to recommend us?",
    "Rate your support experience", "How easy was the setup process?",
    "Overall satisfaction with documentation", "Rate the onboarding experience",
]
SURVEY_SCORES = ["3.2/5", "3.8/5", "4.0/5", "4.3/5", "4.5/5", "4.7/5"]
RESPONSE_COUNTS = ["48 responses", "124 responses", "289 responses", "512 responses", "1,024 responses"]

INVENTORY_ITEMS = [
    ("Widget A-100", "SKU-10042", "342 units"), ("Sensor Module B", "SKU-20081", "89 units"),
    ("Power Supply 12V", "SKU-30015", "1,204 units"), ("Cable Assembly C", "SKU-40027", "567 units"),
    ("Display Panel 7in", "SKU-50033", "23 units"), ("PCB Board Rev3", "SKU-60019", "891 units"),
    ("Battery Pack Li-Ion", "SKU-70044", "156 units"), ("Connector Kit D", "SKU-80011", "2,340 units"),
]

WAREHOUSE_LOCATIONS = ["Warehouse A - Shelf 3", "Warehouse B - Rack 12", "Warehouse C - Bin 7", "Distribution Center"]

DNS_RECORD_TYPES = ["A", "AAAA", "CNAME", "MX", "TXT", "NS"]
DNS_VALUES = [
    "192.168.1.100", "10.0.0.1", "2001:db8::1", "mail.example.com",
    "v=spf1 include:_spf.google.com ~all", "ns1.cloudflare.com",
]
TTL_VALUES = ["300", "3600", "86400"]

BLOG_TITLES = [
    "Why We Rewrote Our Backend in Rust", "Lessons From Scaling to 10M Users",
    "The Hidden Cost of Microservices", "Building a Design System From Scratch",
    "How We Cut Our Cloud Bill by 60%", "Migrating 5TB of Data With Zero Downtime",
    "Our Journey to Zero-Trust Security", "Why We Switched to Server Components",
    "Load Testing: What We Learned the Hard Way", "A Year of Running Kubernetes in Production",
    "Rethinking Our Monorepo Strategy", "Making Accessibility a First-Class Feature",
]

BLOG_TAGS = [
    "engineering", "infrastructure", "frontend", "backend", "devops",
    "security", "performance", "design", "data", "culture", "mobile", "ai",
]

COMMENT_BODIES = [
    "Great post! We ran into the same issue at our company.",
    "Can you elaborate on the migration strategy you used?",
    "This is exactly what I was looking for, thanks for sharing.",
    "We took a different approach but your results are impressive.",
    "How did you handle the rollback plan during the migration?",
    "Interesting. Does this approach scale beyond 100 nodes?",
    "I'd love to see a follow-up on the monitoring setup.",
    "We've been considering the same architecture change.",
]

PLAN_TIERS = [
    {"name": "Free", "price": "$0/mo", "features": "5 users, 1GB storage, email support"},
    {"name": "Starter", "price": "$12/mo", "features": "25 users, 10GB storage, chat support"},
    {"name": "Pro", "price": "$49/mo", "features": "100 users, 100GB storage, priority support"},
    {"name": "Business", "price": "$99/mo", "features": "Unlimited users, 500GB, dedicated CSM"},
    {"name": "Enterprise", "price": "Custom", "features": "Unlimited everything, SLA, on-prem option"},
]

FILE_NAMES = [
    "report-q1-2026.pdf", "architecture-diagram.png", "meeting-notes.md",
    "budget-forecast.xlsx", "api-spec-v3.yaml", "logo-final.svg",
    "deploy-script.sh", "customer-data-export.csv", "design-mockup.fig",
    "test-results.html", "infra-topology.drawio", "backup-config.json",
]

FILE_SIZES = ["12 KB", "48 KB", "156 KB", "1.2 MB", "3.4 MB", "8.7 MB", "24 MB", "67 MB"]
FILE_MODIFIED_DATES = [
    "Mar 10, 2026", "Mar 8, 2026", "Mar 5, 2026", "Feb 28, 2026",
    "Feb 14, 2026", "Jan 20, 2026", "Jan 5, 2026", "Dec 15, 2025",
]

CHAT_MESSAGES = [
    "Hi, I'm having trouble with my account settings.",
    "Can you help me reset my password?",
    "I'm getting a 403 error when accessing the dashboard.",
    "My invoice shows a charge I don't recognize.",
    "The export feature isn't working for me.",
    "How do I add a new team member?",
    "Is there planned maintenance this weekend?",
    "My API key seems to have expired.",
]

CHAT_RESPONSES = [
    "Sure, I can help you with that. Let me look into it.",
    "I've reset your password. You should receive an email shortly.",
    "That error usually means your session expired. Please log in again.",
    "Let me check the billing records for your account.",
    "This is a known issue and we're working on a fix.",
    "You can add team members from Settings > Team > Invite.",
    "Yes, we have scheduled maintenance on Saturday from 2-4 AM UTC.",
    "I've regenerated your API key. Please check your dashboard.",
]

AUDIT_ACTIONS = [
    "user.login", "user.logout", "user.created", "user.deleted",
    "role.changed", "settings.updated", "api_key.rotated", "export.started",
    "deployment.triggered", "webhook.created", "permission.granted", "mfa.enabled",
    "password.changed", "team.member_added", "billing.plan_changed", "token.revoked",
]

AUDIT_IPS = [
    "203.0.113.42", "198.51.100.17", "192.0.2.88", "172.16.0.1",
    "10.0.1.55", "100.64.0.33", "203.0.113.99", "198.51.100.200",
]

SERVER_NAMES = [
    "web-prod-01", "web-prod-02", "api-prod-01", "api-prod-02",
    "db-primary", "db-replica-01", "worker-01", "worker-02",
    "cache-01", "queue-01", "gateway-01", "scheduler-01",
]

CPU_VALUES = ["12%", "28%", "42%", "55%", "68%", "75%", "82%", "91%"]
MEMORY_VALUES = ["2.4 GB / 8 GB", "4.1 GB / 16 GB", "6.8 GB / 32 GB", "12.3 GB / 64 GB", "28.7 GB / 128 GB"]
DISK_VALUES = ["45% used", "62% used", "71% used", "83% used", "89% used"]
UPTIME_VALUES = ["14 days", "28 days", "45 days", "72 days", "120 days", "183 days"]

PIPELINE_NAMES = [
    "build-and-test", "deploy-staging", "deploy-production", "lint-and-format",
    "security-scan", "integration-tests", "e2e-tests", "docker-build",
    "release-candidate", "canary-deploy", "rollback-production", "db-migration",
]

PIPELINE_STATUSES = ["Passed", "Failed", "Running", "Pending", "Cancelled", "Skipped"]
PIPELINE_DURATIONS = ["45s", "1m 23s", "2m 12s", "3m 45s", "5m 30s", "8m 12s", "12m 05s"]
COMMIT_SHAS = ["a1b2c3d", "e4f5g6h", "i7j8k9l", "m0n1o2p", "q3r4s5t", "u6v7w8x"]
BRANCH_NAMES = [
    "main", "feature/auth-v2", "fix/memory-leak", "feature/dashboard-redesign",
    "hotfix/payment-bug", "release/v3.2", "feature/api-v3", "fix/cors-headers",
]

STOCK_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
STOCK_PRICES = ["$142.50", "$175.30", "$198.75", "$224.80", "$267.45", "$312.60", "$456.20", "$892.10"]
STOCK_CHANGES = ["+1.23 (+0.87%)", "+3.45 (+1.98%)", "-2.10 (-1.05%)", "+0.87 (+0.35%)",
                 "-5.60 (-2.14%)", "+8.92 (+2.91%)", "-1.45 (-0.62%)", "+12.30 (+3.45%)"]
MARKET_CAPS = ["$1.2T", "$2.1T", "$850B", "$1.8T", "$620B", "$3.4T", "$540B", "$310B"]
VOLUME_VALUES = ["12.4M", "8.7M", "24.1M", "15.8M", "32.5M", "6.2M", "18.9M"]

HOTEL_NAMES = [
    "The Grand Meridian", "Sunset Harbor Resort", "Alpine View Lodge",
    "City Central Hotel", "Seaside Boutique Inn", "The Regency Suites",
    "Mountain Peak Retreat", "Downtown Plaza Hotel", "Lakeside Manor",
]

ROOM_TYPES = ["Standard Room", "Deluxe Room", "Suite", "Executive Suite", "Penthouse"]
ROOM_RATES = ["$129/night", "$179/night", "$249/night", "$349/night", "$499/night", "$749/night"]
BOOKING_STATUSES = ["Confirmed", "Pending", "Checked In", "Checked Out", "Cancelled"]
BOOKING_IDS = ["BK-88123", "BK-44567", "BK-99012", "BK-12345", "BK-77890", "BK-55678"]

REVIEW_BODIES = [
    "Excellent product, exceeded my expectations. Fast shipping too.",
    "Good quality but the setup instructions could be clearer.",
    "Works as advertised. I've been using it daily for a month now.",
    "Decent for the price. Could use some improvements.",
    "Outstanding customer support when I had an issue.",
    "The build quality is impressive. Highly recommended.",
    "Average experience. Nothing special but gets the job done.",
    "Had some issues initially but support helped me resolve them.",
]

REVIEW_STARS = ["2 stars", "3 stars", "4 stars", "5 stars"]
REVIEW_HELPFULNESS = ["12 found helpful", "34 found helpful", "67 found helpful", "142 found helpful"]
VERIFIED_BADGES = ["Verified Purchase", "Verified Buyer", "Certified Reviewer"]

SETTING_CATEGORIES = [
    ("General", ["Display Name", "Email Address", "Language", "Timezone"]),
    ("Security", ["Two-Factor Authentication", "Session Timeout", "IP Allowlist"]),
    ("Notifications", ["Email Alerts", "Push Notifications", "Weekly Digest"]),
    ("Billing", ["Payment Method", "Invoice Email", "Auto-Renew"]),
    ("API", ["API Key", "Rate Limit Tier", "Webhook URL"]),
    ("Appearance", ["Theme", "Compact Mode", "Sidebar Position"]),
]

SETTING_VALUES = [
    "Enabled", "Disabled", "UTC-8 (Pacific)", "English (US)",
    "Dark", "Light", "Left", "Right", "30 minutes", "1 hour",
    "Standard", "Premium", "Custom",
]

EVENT_TYPES = ["Conference", "Meetup", "Workshop", "Webinar", "Hackathon", "Summit"]
EVENT_VENUES = [
    "Moscone Center, San Francisco", "Convention Center, Austin TX",
    "ExCeL London", "Online (Zoom)", "TechHub Berlin",
    "Javits Center, New York", "Online (Google Meet)", "Pier 48, San Francisco",
]

EVENT_SPEAKERS = [
    f"{n}, {t}" for n, t in zip(
        PERSON_NAMES[:10],
        ["CTO at TechNova", "Staff Engineer at CloudPeak", "VP Engineering at DataForge",
         "Principal Architect at NexGen Labs", "CEO at RedPanda AI",
         "Director of Engineering at Apex Software", "Distinguished Engineer at Helix Solutions",
         "Head of Platform at Velocity Corp", "Founder at BrightEdge", "Senior SRE at Meridian Tech"],
    )
]

ERROR_LEVELS = ["ERROR", "CRITICAL", "WARNING", "FATAL"]
ERROR_MESSAGES = [
    "NullPointerException in UserService.getProfile()",
    "OutOfMemoryError: Java heap space exceeded",
    "ConnectionRefusedError: Cannot connect to database at 10.0.1.5:5432",
    "TimeoutError: Request to auth-service timed out after 30000ms",
    "IndexError: list index out of range in batch_processor.py:142",
    "PermissionError: Access denied to /var/log/app/production.log",
    "ValueError: Invalid JSON payload in webhook handler",
    "SSLError: Certificate expired for api.partner-service.com",
    "RateLimitExceeded: 429 Too Many Requests from IP 203.0.113.42",
    "DiskSpaceError: /data partition 98% full on worker-03",
]

ERROR_STACK_COUNTS = ["seen 3 times", "seen 12 times", "seen 47 times", "seen 128 times", "seen 512 times"]


# ============================================================================
# Page builder functions -- each returns (url, path, html, selector, expected)
# ============================================================================

def _wrap_html(title, body_content, rng=None, nav_items=None, has_footer=True, extra_head=""):
    nav = ""
    if nav_items:
        links = "".join(f'<a href="{href}">{label}</a>' for label, href in nav_items)
        nav = f'<nav class="main-nav" role="navigation">{links}</nav>'
    footer = '<footer><p>Copyright 2025-2026. All rights reserved.</p></footer>' if has_footer else ''

    noise = ""
    if rng and rng.random() < 0.65:
        noise_parts = []
        if rng.random() < 0.4:
            noise_parts.append('<div class="banner-promo" data-campaign="spring26" aria-hidden="true" style="display:none"></div>')
        if rng.random() < 0.4:
            noise_parts.append('<script>window.__CONFIG__={env:"production",version:"3.2.1"}</script>')
        if rng.random() < 0.3:
            noise_parts.append('<noscript><p>Please enable JavaScript to use this application.</p></noscript>')
        if rng.random() < 0.3:
            noise_parts.append('<div id="cookie-consent" style="display:none"><p>We use cookies.</p><button>Accept</button></div>')
        if rng.random() < 0.35:
            noise_parts.append('<script defer src="/analytics.js" data-site-id="UA-12345"></script>')
        if rng.random() < 0.3:
            noise_parts.append('<div class="toast-container" aria-live="polite"></div>')
        if rng.random() < 0.25:
            noise_parts.append('<div id="chat-widget" data-position="bottom-right" style="display:none"><iframe src="about:blank"></iframe></div>')
        if rng.random() < 0.3:
            noise_parts.append('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preload" as="style" href="/assets/main.css">')
        if rng.random() < 0.2:
            noise_parts.append('<svg style="display:none"><symbol id="icon-check"><path d="M5 13l4 4L19 7"/></symbol></svg>')
        if rng.random() < 0.25:
            noise_parts.append('<div class="skip-link"><a href="#main-content">Skip to content</a></div>')
        noise = "".join(noise_parts)

    style = ""
    if rng and rng.random() < 0.5:
        style_choices = [
            '<style>body{font-family:system-ui,sans-serif;margin:0;padding:0}main{max-width:1200px;margin:0 auto}</style>',
            '<style>*{box-sizing:border-box}body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;color:#1a1a2e;background:#fafafa}</style>',
            '<style>:root{--primary:#2563eb;--bg:#fff;--text:#111}body{font-family:Inter,sans-serif;margin:0}a{color:var(--primary)}</style>',
        ]
        style = rng.choice(style_choices)

    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{title}</title>{style}{extra_head}</head>'
        f'<body>{noise}{nav}<main>{body_content}</main>{footer}</body></html>'
    )


def build_product_page(rng):
    company = rng.choice(COMPANY_NAMES)
    product = rng.choice(PRODUCT_NAMES)
    price = rng.choice(PRICES)
    rating = rng.choice(RATINGS)
    reviews = rng.choice(REVIEW_COUNTS)
    stock = rng.choice(STOCK_STATUSES + OUT_OF_STOCK)
    slug = product.lower().replace(" ", "-").replace("'", "")
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="product"><h1 class="product-name">{product}</h1>'
        f'<div class="pricing"><span id="product-price">{price}</span>'
        f'<span class="old-price">{rng.choice(PRICES)}</span></div>'
        f'<div class="availability"><span class="stock-status">{stock}</span></div>'
        f'<div class="reviews"><span data-testid="avg-rating">{rating}</span>'
        f'<span class="review-count">({reviews})</span></div>'
        f'<div class="description"><p>High-quality {product.lower()} from {company}.</p></div></div>'
    )
    html = _wrap_html(f"{product} - {company}", body,
                       rng=rng, nav_items=[("Home", "/"), ("Products", "/products")])

    targets = [
        ("#product-price", price),
        (".stock-status", stock),
        ('[data-testid="avg-rating"]', rating),
        (".product-name", product),
        (".review-count", f"({reviews})"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/products/{slug}"
    return url, html, selector, expected


def build_article_page(rng):
    title = rng.choice(ARTICLE_TITLES)
    author = rng.choice(PERSON_NAMES)
    date = rng.choice(DATES)
    domain = rng.choice(DOMAINS)
    slug = title.lower().replace(" ", "-").replace("'", "")[:40]

    body = (
        f'<article><h1 class="article-title">{title}</h1>'
        f'<div class="article-meta"><span class="author">By {author}</span>'
        f'<time datetime="2026-01-15">{date}</time></div>'
        f'<div class="article-body"><p>An in-depth look at {title.lower()} and what it means for the industry. '
        f'Experts predict significant changes ahead as technology continues to evolve.</p>'
        f'<p>Industry analysts have noted that this trend accelerated dramatically in the past year, '
        f'with adoption rates increasing by over 40% across major markets.</p></div></article>'
        f'<aside class="sidebar"><h3>Related Posts</h3><ul><li><a href="/posts/related-1">More on this topic</a></li></ul></aside>'
    )
    html = _wrap_html(f"{title} - Tech News", body,
                       rng=rng, nav_items=[("Home", "/"), ("Articles", "/articles"), ("About", "/about")])

    targets = [
        (".article-title", title),
        (".author", f"By {author}"),
        (".article-body p", title.lower()),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/articles/{slug}"
    return url, html, selector, expected


def build_dashboard_page(rng):
    company = rng.choice(COMPANY_NAMES)
    users = rng.choice(METRIC_VALUES_USERS)
    revenue = rng.choice(METRIC_VALUES_REVENUE)
    conv = rng.choice(CONVERSION_RATES)
    sessions = rng.choice(SESSION_COUNTS)
    domain = rng.choice(DOMAINS)

    body = (
        f'<h1>Dashboard Overview</h1>'
        f'<div class="kpi-grid">'
        f'<div class="kpi-card"><h3>Total Users</h3><span class="kpi-value" id="total-users">{users}</span></div>'
        f'<div class="kpi-card"><h3>Revenue (MTD)</h3><span class="kpi-value" id="revenue-mtd">{revenue}</span></div>'
        f'<div class="kpi-card"><h3>Active Sessions</h3><span class="kpi-value" id="active-sessions">{sessions}</span></div>'
        f'<div class="kpi-card"><h3>Conversion Rate</h3><span class="kpi-value" id="conversion-rate">{conv}</span></div>'
        f'</div><div class="chart-placeholder"><p>Chart data loading...</p></div>'
    )
    html = _wrap_html(f"Dashboard - {company}", body,
                       rng=rng, nav_items=[("Dashboard", "/dashboard"), ("Reports", "/reports"), ("Settings", "/settings")])

    targets = [
        ("#total-users", users),
        ("#revenue-mtd", revenue),
        ("#conversion-rate", conv),
        ("#active-sessions", sessions),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/dashboard"
    return url, html, selector, expected


def build_status_page(rng):
    company = rng.choice(COMPANY_NAMES)
    services = rng.sample(["API Gateway", "Database Cluster", "CDN", "Auth Service",
                           "Search Engine", "Payment Processor", "Email Service",
                           "Storage", "Load Balancer", "Cache Layer"], k=rng.randint(3, 5))
    statuses = [rng.choice(STATUS_OPERATIONAL + STATUS_DEGRADED) for _ in services]
    has_issue = any(s in STATUS_DEGRADED for s in statuses)
    overall = "Some systems experiencing issues" if has_issue else "All systems operational"
    domain = rng.choice(DOMAINS)

    rows = "".join(
        f'<div class="service-row"><span class="service-name">{svc}</span>'
        f'<span class="service-status">{st}</span></div>'
        for svc, st in zip(services, statuses)
    )
    body = (
        f'<h1>{company} Status</h1>'
        f'<p class="last-updated">Last updated: 2026-03-10 14:30 UTC</p>'
        f'<div class="services-list">{rows}</div>'
        f'<div id="overall-status" role="alert"><strong>{overall}</strong></div>'
    )
    html = _wrap_html(f"Status - {company}", body, rng=rng)

    targets = [
        ("#overall-status", overall),
        ('[role="alert"]', overall),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://status.{domain}/services"
    return url, html, selector, expected


def build_incident_page(rng):
    company = rng.choice(COMPANY_NAMES)
    inc_title = rng.choice(INCIDENT_TITLES)
    severity = rng.choice(SEVERITIES)
    domain = rng.choice(DOMAINS)
    inc_id = f"INC-{rng.randint(1000,9999)}"

    body = (
        f'<div class="incident"><h2 class="incident-title">{inc_title}</h2>'
        f'<div class="incident-meta"><span class="severity-badge" id="severity">{severity}</span>'
        f'<span class="incident-time">Started: 2026-03-10 12:15 UTC</span></div>'
        f'<div class="incident-updates"><div class="update"><time>14:30 UTC</time>'
        f'<p class="update-text">Engineering team is actively working on a resolution.</p></div></div></div>'
    )
    html = _wrap_html(f"Incident {inc_id} - {company}", body, rng=rng)
    targets = [
        (".severity-badge", severity),
        ("#severity", severity),
        (".incident-title", inc_title),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://status.{domain}/incidents/{inc_id}"
    return url, html, selector, expected


def build_profile_page(rng):
    name = rng.choice(PERSON_NAMES)
    email = name.lower().replace(" ", ".") + "@" + rng.choice(DOMAINS)
    role = rng.choice(["Administrator", "Editor", "Viewer", "Manager", "Developer", "Analyst"])
    domain = rng.choice(DOMAINS)
    username = name.lower().replace(" ", "")

    body = (
        f'<div class="profile-card"><h2 class="display-name">{name}</h2>'
        f'<p id="user-email">{email}</p>'
        f'<div class="profile-details"><span class="role" id="user-role">{role}</span>'
        f'<span class="joined">Member since Jan 2024</span></div></div>'
    )
    html = _wrap_html(f"Profile - {name}", body,
                       rng=rng, nav_items=[("Dashboard", "/dashboard"), ("Profile", "/profile"), ("Settings", "/settings")])

    targets = [
        ("#user-email", email),
        ("#user-role", role),
        (".display-name", name),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/profile/{username}"
    return url, html, selector, expected


def build_ticket_page(rng):
    tid = rng.choice(TICKET_IDS)
    subject = rng.choice(TICKET_SUBJECTS)
    priority = rng.choice(PRIORITIES)
    status = rng.choice(TICKET_STATUSES)
    assignee = rng.choice(PERSON_NAMES)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="ticket"><h1>{tid}: {subject}</h1>'
        f'<div class="ticket-meta"><span class="ticket-priority" id="ticket-priority">{priority}</span>'
        f'<span class="ticket-status" id="ticket-status">{status}</span>'
        f'<span class="ticket-assignee">Assigned to: {assignee}</span></div>'
        f'<div class="ticket-body"><p>Users report that {subject.lower()}. This is affecting production workloads.</p></div></div>'
    )
    html = _wrap_html(f"Ticket {tid}", body, rng=rng, nav_items=[("All Tickets", "/tickets")])

    targets = [
        ("#ticket-status", status),
        ("#ticket-priority", priority),
        (".ticket-assignee", f"Assigned to: {assignee}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://support.{domain}/tickets/{tid}"
    return url, html, selector, expected


def build_weather_page(rng):
    city = rng.choice(CITIES)
    temp = rng.choice(TEMPERATURES)
    condition = rng.choice(CONDITIONS)
    domain = rng.choice(DOMAINS)
    slug = city.split(",")[0].lower().replace(" ", "-")

    body = (
        f'<div class="weather-card"><h1>{city}</h1>'
        f'<div class="current-weather"><span class="temperature" id="current-temp">{temp}</span>'
        f'<span class="condition" id="weather-condition">{condition}</span></div>'
        f'<div class="forecast"><h3>5-Day Forecast</h3>'
        f'<div class="forecast-day"><span class="day">Mon</span><span class="high">{rng.choice(TEMPERATURES)}</span></div>'
        f'<div class="forecast-day"><span class="day">Tue</span><span class="high">{rng.choice(TEMPERATURES)}</span></div></div></div>'
    )
    html = _wrap_html(f"Weather - {city}", body, rng=rng)

    targets = [
        ("#current-temp", temp),
        ("#weather-condition", condition),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://weather.{domain}/city/{slug}"
    return url, html, selector, expected


def build_job_page(rng):
    title = rng.choice(JOB_TITLES)
    dept = rng.choice(DEPARTMENTS)
    location = rng.choice(CITIES)
    salary = rng.choice(SALARY_RANGES)
    company = rng.choice(COMPANY_NAMES)
    domain = rng.choice(DOMAINS)
    slug = title.lower().replace(" ", "-")

    body = (
        f'<div class="job-posting"><h1 class="job-title">{title}</h1>'
        f'<div class="job-meta"><span class="department">{dept}</span>'
        f'<span class="location" id="job-location">{location} (Hybrid)</span>'
        f'<span class="salary-range" id="salary-range">{salary}</span></div>'
        f'<div class="job-description"><h2>About the Role</h2>'
        f'<p>Join {company} as a {title}. Work with a world-class team on cutting-edge technology.</p></div>'
        f'<div class="job-requirements"><h2>Requirements</h2><ul>'
        f'<li>5+ years of relevant experience</li><li>Strong communication skills</li></ul></div></div>'
    )
    html = _wrap_html(f"{title} - {company} Careers", body, rng=rng, nav_items=[("All Jobs", "/jobs")])

    targets = [
        ("#salary-range", salary),
        ("#job-location", f"{location} (Hybrid)"),
        (".job-title", title),
        (".department", dept),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://careers.{domain}/jobs/{slug}"
    return url, html, selector, expected


def build_billing_page(rng):
    plan = rng.choice(PLAN_NAMES[1:4])
    price = rng.choice(["$9/mo", "$29/mo", "$49/mo", "$79/mo", "$149/mo"])
    card_last4 = str(rng.randint(1000, 9999))
    next_date = rng.choice(["April 1, 2026", "April 15, 2026", "May 1, 2026"])
    domain = rng.choice(DOMAINS)

    body = (
        f'<h1>Billing & Subscription</h1>'
        f'<div class="plan-info"><span class="plan-name">{plan} Plan</span>'
        f'<span id="monthly-cost">{price}</span></div>'
        f'<div class="payment-method"><h3>Payment Method</h3>'
        f'<span id="card-info">Visa ending in {card_last4}</span></div>'
        f'<div class="next-billing"><span id="next-bill-date">Next billing: {next_date}</span></div>'
    )
    html = _wrap_html(f"Billing - {domain}", body,
                       rng=rng, nav_items=[("General", "/settings/general"), ("Billing", "/settings/billing")])

    targets = [
        ("#monthly-cost", price),
        ("#card-info", f"Visa ending in {card_last4}"),
        ("#next-bill-date", f"Next billing: {next_date}"),
        (".plan-name", f"{plan} Plan"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/settings/billing"
    return url, html, selector, expected


def build_flight_page(rng):
    code = rng.choice(FLIGHT_CODES)
    orig = rng.choice(AIRPORTS)
    dest = rng.choice([a for a in AIRPORTS if a != orig])
    status = rng.choice(FLIGHT_STATUSES)
    gate = rng.choice(GATES)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="flight-info"><h1>Flight {code}</h1>'
        f'<div class="route"><span class="origin">{orig[0]} - {orig[1]}</span>'
        f'<span class="arrow">→</span><span class="destination">{dest[0]} - {dest[1]}</span></div>'
        f'<div class="status-section"><span id="flight-status">{status}</span>'
        f'<span class="gate" id="gate-info">Gate: {gate}</span></div></div>'
    )
    html = _wrap_html(f"Flight {code} Status", body, rng=rng)

    targets = [
        ("#flight-status", status),
        ("#gate-info", f"Gate: {gate}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://flights.{domain}/status/{code}"
    return url, html, selector, expected


def build_banking_page(rng):
    acct_type = rng.choice(ACCOUNT_TYPES)
    balance = rng.choice(BALANCE_AMOUNTS)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="account-summary"><h1>{acct_type}</h1>'
        f'<div class="balance-section"><span class="label">Available Balance</span>'
        f'<span id="account-balance" class="balance-amount">{balance}</span></div>'
        f'<div class="recent-transactions"><h2>Recent Transactions</h2>'
        f'<table><tbody><tr><td>Mar 8</td><td>Grocery Store</td><td class="debit">-$62.40</td></tr>'
        f'<tr><td>Mar 7</td><td>Direct Deposit</td><td class="credit">+$3,200.00</td></tr></tbody></table></div></div>'
    )
    html = _wrap_html(f"{acct_type} - Secure Banking", body,
                       rng=rng, nav_items=[("Accounts", "/accounts"), ("Transfers", "/transfers")])

    targets = [("#account-balance", balance)]
    selector, expected = rng.choice(targets)
    url = f"https://banking.{domain}/accounts/{acct_type.lower().replace(' ', '-')}"
    return url, html, selector, expected


def build_course_page(rng):
    code = rng.choice(COURSE_CODES)
    name = rng.choice(COURSE_NAMES)
    instructor = rng.choice(INSTRUCTORS)
    enrolled = rng.randint(20, 95)
    capacity = rng.choice([50, 80, 100, 120])
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="course"><h1 class="course-title">{code}: {name}</h1>'
        f'<div class="course-meta"><span id="instructor">Instructor: {instructor}</span>'
        f'<span class="schedule">Mon/Wed 2:00-3:30 PM</span></div>'
        f'<div class="enrollment"><span id="enrollment-status">Enrollment: {enrolled}/{capacity} seats filled</span></div>'
        f'<div class="course-desc"><p>A comprehensive course covering key topics in {name.lower()}.</p></div></div>'
    )
    html = _wrap_html(f"{code} - University", body, rng=rng)

    targets = [
        ("#instructor", f"Instructor: {instructor}"),
        ("#enrollment-status", f"Enrollment: {enrolled}/{capacity} seats filled"),
        (".course-title", f"{code}: {name}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://university.{domain}/courses/{code}"
    return url, html, selector, expected


def build_tracking_page(rng):
    tid = rng.choice(TRACKING_IDS)
    status = rng.choice(DELIVERY_STATUSES)
    eta = rng.choice(DELIVERY_ETAS)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="tracking"><h1>Package {tid}</h1>'
        f'<div class="tracking-summary"><span id="delivery-status" class="status">{status}</span>'
        f'<span class="eta" id="delivery-eta">Expected: {eta}</span></div>'
        f'<div class="tracking-history"><div class="event"><time>Mar 10, 8:30 AM</time><p>{status}</p></div>'
        f'<div class="event"><time>Mar 9, 6:00 PM</time><p>In transit</p></div></div></div>'
    )
    html = _wrap_html(f"Track {tid}", body, rng=rng)

    targets = [
        ("#delivery-status", status),
        ("#delivery-eta", f"Expected: {eta}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://tracking.{domain}/track/{tid}"
    return url, html, selector, expected


def build_movie_page(rng):
    title = rng.choice(MOVIE_TITLES)
    critic = rng.choice(MOVIE_SCORES)
    audience = rng.choice(MOVIE_SCORES)
    domain = rng.choice(DOMAINS)
    slug = title.lower().replace(" ", "-").replace(":", "")[:30]

    body = (
        f'<div class="movie"><h1 class="movie-title">{title}</h1>'
        f'<div class="movie-meta"><span class="year">2026</span><span class="genre">Sci-Fi, Drama</span></div>'
        f'<div class="ratings"><span class="critics-score" id="critics-score">{critic}</span>'
        f'<span class="audience-score" id="audience-score">{audience}</span></div>'
        f'<p class="synopsis">A compelling story that pushes the boundaries of the genre.</p></div>'
    )
    html = _wrap_html(f"{title} - Movie Reviews", body, rng=rng)

    targets = [
        ("#critics-score", critic),
        ("#audience-score", audience),
        (".movie-title", title),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://movies.{domain}/film/{slug}"
    return url, html, selector, expected


def build_restaurant_page(rng):
    items = rng.sample(CUISINE_ITEMS, k=rng.randint(3, 5))
    domain = rng.choice(DOMAINS)
    target_item = rng.choice(items)
    item_id = target_item[0].lower().replace(" ", "-")

    menu_rows = "".join(
        f'<div class="menu-item"><span class="item-name">{name}</span>'
        f'<span class="item-price" id="{name.lower().replace(" ", "-")}-price">{price}</span></div>'
        for name, price in items
    )
    body = f'<h1>Our Menu</h1><div class="menu-section"><h2>Main Courses</h2>{menu_rows}</div>'
    html = _wrap_html("Menu - Restaurant", body, rng=rng)

    selector = f"#{item_id}-price"
    expected = target_item[1]
    url = f"https://order.{domain}/menu"
    return url, html, selector, expected


def build_podcast_page(rng):
    title = rng.choice(PODCAST_TITLES)
    ep_num = rng.randint(10, 300)
    duration = rng.choice(DURATIONS)
    plays = rng.choice(PLAY_COUNTS)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="episode"><h1 class="episode-title">Episode {ep_num}: {title}</h1>'
        f'<div class="episode-meta"><span class="show-name">Tech Talks Podcast</span>'
        f'<span class="publish-date">Published: {rng.choice(DATES)}</span>'
        f'<span id="episode-duration">Duration: {duration}</span></div>'
        f'<div class="episode-desc"><p>In this episode, we explore {title.lower()} and discuss practical implications.</p></div>'
        f'<div class="listen-stats"><span id="play-count">{plays} plays</span></div></div>'
    )
    html = _wrap_html(f"Ep {ep_num} - Tech Talks", body, rng=rng)

    targets = [
        ("#episode-duration", f"Duration: {duration}"),
        ("#play-count", f"{plays} plays"),
        (".episode-title", f"Episode {ep_num}: {title}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://podcasts.{domain}/show/tech-talks/ep-{ep_num}"
    return url, html, selector, expected


def build_leaderboard_page(rng):
    players = rng.sample(PLAYER_NAMES, k=min(5, len(PLAYER_NAMES)))
    scores = sorted(rng.sample(SCORES, k=len(players)), key=lambda s: int(s.replace(",", "")), reverse=True)
    domain = rng.choice(DOMAINS)

    rows = "".join(
        f'<tr class="rank-{i+1}"><td>{i+1}</td><td class="player-name">{p}</td>'
        f'<td class="score" id="score-rank-{i+1}">{s}</td></tr>'
        for i, (p, s) in enumerate(zip(players, scores))
    )
    body = (
        f'<h1>Global Leaderboard</h1><table class="leaderboard">'
        f'<thead><tr><th>Rank</th><th>Player</th><th>Score</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )
    html = _wrap_html("Leaderboard - Arena", body, rng=rng)

    rank = rng.randint(1, len(players))
    selector = f"#score-rank-{rank}"
    expected = scores[rank - 1]
    url = f"https://game.{domain}/leaderboard"
    return url, html, selector, expected


def build_realty_page(rng):
    listing_type = rng.choice(LISTING_TYPES)
    price = rng.choice(LISTING_PRICES)
    neighborhood = rng.choice(NEIGHBORHOODS)
    city = rng.choice(CITIES)
    domain = rng.choice(DOMAINS)
    mls = f"MLS-{rng.randint(100000, 999999)}"

    body = (
        f'<div class="listing"><h1 class="listing-title">Beautiful {listing_type} in {neighborhood}</h1>'
        f'<div class="listing-meta"><span id="listing-price" class="price">{price}</span>'
        f'<span class="beds">{rng.randint(1,5)} Beds</span><span class="baths">{rng.randint(1,3)} Baths</span></div>'
        f'<div class="listing-details"><p class="address" id="listing-address">{neighborhood}, {city}</p>'
        f'<p class="description">Stunning property with modern finishes and great location.</p></div></div>'
    )
    html = _wrap_html(f"{listing_type} in {neighborhood} - Realty", body, rng=rng)

    targets = [
        ("#listing-price", price),
        ("#listing-address", f"{neighborhood}, {city}"),
        (".listing-title", f"Beautiful {listing_type} in {neighborhood}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://homes.{domain}/listing/{mls}"
    return url, html, selector, expected


def build_email_page(rng):
    sender = rng.choice(EMAIL_SENDERS)
    subject = rng.choice(EMAIL_SUBJECTS)
    unread = rng.randint(1, 25)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="inbox"><h1>Inbox</h1><div id="unread-badge" class="badge">{unread} unread</div>'
        f'<div class="email-list"><div class="email unread"><span class="sender">{sender}</span>'
        f'<span class="subject">{subject}</span><span class="date">Mar 10</span></div></div></div>'
    )
    html = _wrap_html("Inbox - Mail", body,
                       rng=rng, nav_items=[("Inbox", "/inbox"), ("Sent", "/sent"), ("Drafts", "/drafts")])

    targets = [
        ("#unread-badge", f"{unread} unread"),
        (".sender", sender),
        (".subject", subject),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://mail.{domain}/inbox"
    return url, html, selector, expected


def build_thermostat_page(rng):
    current = rng.choice(TEMPS)
    target = rng.choice(TEMPS)
    mode = rng.choice(HVAC_MODES)
    domain = rng.choice(DOMAINS)

    body = (
        f'<div class="device-panel"><h1>Living Room Thermostat</h1>'
        f'<div class="current-reading"><span class="label">Current Temperature</span>'
        f'<span id="current-temp" class="temp-value">{current}</span></div>'
        f'<div class="target-setting"><span class="label">Target</span>'
        f'<span id="target-temp" class="temp-value">{target}</span></div>'
        f'<div class="mode"><span id="hvac-mode">Mode: {mode}</span></div></div>'
    )
    html = _wrap_html("Thermostat - Smart Home", body, rng=rng)

    targets = [
        ("#current-temp", current),
        ("#target-temp", target),
        ("#hvac-mode", f"Mode: {mode}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://smarthome.{domain}/devices/thermostat"
    return url, html, selector, expected


def build_forum_page(rng):
    topic = rng.choice(FORUM_TOPICS)
    author = rng.choice(PERSON_NAMES)
    replies = rng.choice(REPLY_COUNTS)
    views = rng.choice(VIEW_COUNTS)
    domain = rng.choice(DOMAINS)
    tid = rng.randint(10000, 99999)
    last_reply = rng.choice(PERSON_NAMES)

    reply_items = "".join(
        f'<div class="reply" data-reply-id="{i}"><div class="reply-author">{rng.choice(PERSON_NAMES)}</div>'
        f'<div class="reply-body"><p>This is a great point about {topic.lower().split()[0]}. '
        f'I would also suggest looking into the documentation.</p></div>'
        f'<div class="reply-meta"><time>{rng.choice(DATES)}</time></div></div>'
        for i in range(rng.randint(2, 4))
    )
    body = (
        f'<div class="thread"><h1 class="thread-title">{topic}</h1>'
        f'<div class="thread-meta"><span class="thread-author">Posted by {author}</span>'
        f'<span id="reply-count">{replies}</span><span id="view-count">{views}</span></div>'
        f'<div class="thread-body"><p>I have been looking into {topic.lower()} and wanted to share my findings. '
        f'Has anyone else encountered similar challenges?</p></div>'
        f'<div class="replies-section"><h2>Replies</h2>{reply_items}</div>'
        f'<div class="thread-footer"><span class="last-reply">Last reply by {last_reply}</span></div></div>'
    )
    html = _wrap_html(f"{topic} - Forum", body, rng=rng,
                       nav_items=[("Home", "/"), ("Forums", "/forums"), ("Search", "/search")])

    targets = [
        ("#reply-count", replies),
        ("#view-count", views),
        (".thread-title", topic),
        (".thread-author", f"Posted by {author}"),
        (".last-reply", f"Last reply by {last_reply}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://forums.{domain}/thread/{tid}"
    return url, html, selector, expected


def build_changelog_page(rng):
    version = rng.choice(CHANGELOG_VERSIONS)
    date = rng.choice(DATES)
    items = rng.sample(CHANGELOG_ITEMS, k=rng.randint(3, 6))
    domain = rng.choice(DOMAINS)

    li_items = "".join(f"<li>{item}</li>" for item in items)
    prev_ver = rng.choice([v for v in CHANGELOG_VERSIONS if v != version])
    body = (
        f'<div class="changelog">'
        f'<h1>Changelog</h1>'
        f'<div class="release" data-version="{version}">'
        f'<h2 class="version-heading" id="latest-version">{version}</h2>'
        f'<p class="release-date" id="release-date">Released: {date}</p>'
        f'<ul class="change-list">{li_items}</ul></div>'
        f'<div class="release" data-version="{prev_ver}">'
        f'<h2 class="version-heading">{prev_ver}</h2>'
        f'<p class="release-date">Released: {rng.choice(DATES)}</p>'
        f'<ul class="change-list"><li>Bug fixes and improvements</li></ul></div></div>'
    )
    html = _wrap_html(f"Changelog - {domain}", body, rng=rng,
                       nav_items=[("Docs", "/docs"), ("Changelog", "/changelog"), ("API", "/api")])

    targets = [
        ("#latest-version", version),
        ("#release-date", f"Released: {date}"),
        (".change-list li", items[0]),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://docs.{domain}/changelog"
    return url, html, selector, expected


def build_search_results_page(rng):
    query = rng.choice(SEARCH_QUERIES)
    result_count = rng.choice(["142", "1,280", "3,420", "8,901", "24,500"])
    domain = rng.choice(DOMAINS)

    results = []
    for i in range(rng.randint(3, 5)):
        tmpl = rng.choice(SEARCH_RESULT_TITLES)
        t = tmpl.format(q=query.title())
        results.append((t, f"https://example.com/result-{i}", f"A comprehensive resource covering {query} with examples and best practices."))

    result_html = "".join(
        f'<div class="search-result" data-position="{i+1}">'
        f'<h3 class="result-title"><a href="{url}">{t}</a></h3>'
        f'<p class="result-snippet">{desc}</p>'
        f'<span class="result-url">{url}</span></div>'
        for i, (t, url, desc) in enumerate(results)
    )
    body = (
        f'<div class="search-page">'
        f'<div class="search-bar"><input type="text" value="{query}" />'
        f'<button>Search</button></div>'
        f'<div class="search-meta"><span id="result-count">About {result_count} results</span>'
        f'<span class="search-time">(0.42 seconds)</span></div>'
        f'<div class="search-results">{result_html}</div>'
        f'<div class="pagination"><a href="?page=2">Next</a></div></div>'
    )
    html = _wrap_html(f"{query} - Search Results", body, rng=rng)

    targets = [
        ("#result-count", f"About {result_count} results"),
        ('.search-result[data-position="1"] .result-title', results[0][0]),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://search.{domain}/q?query={query.replace(' ', '+')}"
    return url, html, selector, expected


def build_notification_page(rng):
    domain = rng.choice(DOMAINS)
    notifs = []
    for _ in range(rng.randint(3, 6)):
        person = rng.choice(PERSON_NAMES)
        action, target = rng.choice(NOTIFICATION_TYPES)
        notifs.append((person, action, target))

    unread = rng.randint(1, len(notifs))
    notif_html = "".join(
        f'<div class="notification{"" if i >= unread else " unread"}" data-id="notif-{i}">'
        f'<span class="notif-actor">{person}</span> '
        f'<span class="notif-action">{action}</span> '
        f'<span class="notif-target">{target}</span>'
        f'<time class="notif-time">{rng.choice(["2 min ago", "15 min ago", "1 hour ago", "3 hours ago", "yesterday"])}</time></div>'
        for i, (person, action, target) in enumerate(notifs)
    )
    body = (
        f'<div class="notifications-page">'
        f'<h1>Notifications</h1>'
        f'<div class="notif-header"><span id="unread-count">{unread} unread</span>'
        f'<button class="mark-all-read">Mark all as read</button></div>'
        f'<div class="notification-list">{notif_html}</div></div>'
    )
    html = _wrap_html("Notifications", body, rng=rng,
                       nav_items=[("Dashboard", "/"), ("Notifications", "/notifications")])

    person0, action0, target0 = notifs[0]
    targets = [
        ("#unread-count", f"{unread} unread"),
        ('.notification[data-id="notif-0"] .notif-actor', person0),
        ('.notification[data-id="notif-0"] .notif-target', target0),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/notifications"
    return url, html, selector, expected


def build_calendar_page(rng):
    event = rng.choice(EVENT_NAMES)
    time_range = rng.choice(EVENT_TIMES)
    location = rng.choice(EVENT_LOCATIONS)
    organizer = rng.choice(PERSON_NAMES)
    date = rng.choice(DATES)
    domain = rng.choice(DOMAINS)
    event_id = f"evt-{rng.randint(1000, 9999)}"

    attendees = rng.sample(PERSON_NAMES, k=rng.randint(2, 5))
    attendee_items = "".join(f'<li class="attendee">{a}</li>' for a in attendees)

    body = (
        f'<div class="event-detail" data-event-id="{event_id}">'
        f'<h1 class="event-name">{event}</h1>'
        f'<div class="event-meta">'
        f'<span id="event-date">{date}</span>'
        f'<span id="event-time">{time_range}</span>'
        f'<span id="event-location">Location: {location}</span></div>'
        f'<div class="event-organizer"><span id="organizer">Organized by {organizer}</span></div>'
        f'<div class="attendee-list"><h3>Attendees ({len(attendees)})</h3>'
        f'<ul>{attendee_items}</ul></div>'
        f'<div class="event-actions"><button>Accept</button><button>Decline</button></div></div>'
    )
    html = _wrap_html(f"{event} - Calendar", body, rng=rng,
                       nav_items=[("Calendar", "/calendar"), ("Events", "/events")])

    targets = [
        ("#event-time", time_range),
        ("#event-location", f"Location: {location}"),
        ("#organizer", f"Organized by {organizer}"),
        (".event-name", event),
        ("#event-date", date),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://calendar.{domain}/event/{event_id}"
    return url, html, selector, expected


def build_cart_page(rng):
    n_items = rng.randint(2, 4)
    cart_items = rng.sample(PRODUCT_NAMES, k=n_items)
    cart_prices = [rng.choice(PRICES) for _ in cart_items]
    cart_qtys = [rng.choice(CART_QUANTITIES) for _ in cart_items]
    total = rng.choice(CART_TOTALS)
    domain = rng.choice(DOMAINS)
    has_promo = rng.random() < 0.4
    promo = rng.choice(PROMO_CODES) if has_promo else None
    discount = rng.choice(DISCOUNT_AMOUNTS) if has_promo else None

    items_html = "".join(
        f'<div class="cart-item" data-item-idx="{i}">'
        f'<span class="item-name">{name}</span>'
        f'<span class="item-qty">Qty: {qty}</span>'
        f'<span class="item-price">{price}</span></div>'
        for i, (name, price, qty) in enumerate(zip(cart_items, cart_prices, cart_qtys))
    )
    promo_html = (
        f'<div class="promo-applied"><span id="promo-code">Code: {promo}</span>'
        f'<span id="discount-amount">{discount}</span></div>'
        if has_promo else ''
    )
    body = (
        f'<div class="shopping-cart"><h1>Shopping Cart</h1>'
        f'<div class="cart-items">{items_html}</div>'
        f'{promo_html}'
        f'<div class="cart-summary"><span class="item-count">{n_items} items</span>'
        f'<span id="cart-total" class="total-amount">Total: {total}</span></div>'
        f'<div class="cart-actions"><button class="checkout-btn">Proceed to Checkout</button></div></div>'
    )
    html = _wrap_html(f"Cart - {domain}", body, rng=rng)

    targets = [
        ("#cart-total", f"Total: {total}"),
        (".item-count", f"{n_items} items"),
        ('.cart-item[data-item-idx="0"] .item-name', cart_items[0]),
    ]
    if has_promo:
        targets.append(("#promo-code", f"Code: {promo}"))
        targets.append(("#discount-amount", discount))
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/cart"
    return url, html, selector, expected


def build_api_docs_page(rng):
    endpoint = rng.choice(API_ENDPOINTS)
    method, path_str, desc = endpoint
    rate_limit = rng.choice(RATE_LIMITS)
    domain = rng.choice(DOMAINS)
    company = rng.choice(COMPANY_NAMES)

    params_html = ""
    if "{id}" in path_str:
        params_html = ('<div class="param"><code class="param-name">id</code>'
                       '<span class="param-type">string</span>'
                       '<span class="param-required">Required</span></div>')

    body = (
        f'<div class="api-doc">'
        f'<div class="api-header"><span class="http-method method-{method.lower()}" id="http-method">{method}</span>'
        f'<code class="endpoint-path" id="endpoint-path">{path_str}</code></div>'
        f'<p class="api-description" id="api-description">{desc}</p>'
        f'<div class="api-meta"><span id="rate-limit">Rate limit: {rate_limit}</span>'
        f'<span class="auth-required">Authentication: Bearer token</span></div>'
        f'{params_html}'
        f'<div class="response-example"><h3>Response</h3>'
        f'<pre><code>{{"status": "ok", "data": []}}</code></pre></div></div>'
    )
    html = _wrap_html(f"API - {method} {path_str} - {company}", body, rng=rng,
                       nav_items=[("Overview", "/api"), ("Endpoints", "/api/endpoints"), ("Auth", "/api/auth")])

    targets = [
        ("#http-method", method),
        ("#endpoint-path", path_str),
        ("#api-description", desc),
        ("#rate-limit", f"Rate limit: {rate_limit}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://api.{domain}/docs{path_str.replace('{id}', '123')}"
    return url, html, selector, expected


def build_recipe_page(rng):
    name = rng.choice(RECIPE_NAMES)
    prep = rng.choice(PREP_TIMES)
    cook = rng.choice(COOK_TIMES)
    servings = rng.choice(RECIPE_SERVINGS)
    rating = rng.choice(RECIPE_RATINGS)
    difficulty = rng.choice(DIFFICULTY_LEVELS)
    author = rng.choice(PERSON_NAMES)
    domain = rng.choice(DOMAINS)
    slug = name.lower().replace(" ", "-").replace("'", "")[:30]

    body = (
        f'<div class="recipe" itemscope itemtype="http://schema.org/Recipe">'
        f'<h1 class="recipe-name" itemprop="name">{name}</h1>'
        f'<div class="recipe-meta">'
        f'<span class="recipe-author">By {author}</span>'
        f'<span id="recipe-rating">{rating}</span>'
        f'<span id="difficulty-level">{difficulty}</span></div>'
        f'<div class="time-info">'
        f'<span id="prep-time">Prep: {prep}</span>'
        f'<span id="cook-time">Cook: {cook}</span>'
        f'<span id="servings">{servings}</span></div>'
        f'<div class="ingredients"><h2>Ingredients</h2>'
        f'<ul><li>2 cups flour</li><li>1 tsp salt</li><li>3 eggs</li></ul></div>'
        f'<div class="instructions"><h2>Instructions</h2>'
        f'<ol><li>Preheat oven to 375F.</li><li>Mix dry ingredients.</li><li>Combine and bake.</li></ol></div></div>'
    )
    html = _wrap_html(f"{name} - Recipes", body, rng=rng,
                       nav_items=[("Home", "/"), ("Recipes", "/recipes"), ("Favorites", "/favorites")])

    targets = [
        ("#recipe-rating", rating),
        ("#prep-time", f"Prep: {prep}"),
        ("#cook-time", f"Cook: {cook}"),
        ("#servings", servings),
        ("#difficulty-level", difficulty),
        (".recipe-name", name),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://recipes.{domain}/recipe/{slug}"
    return url, html, selector, expected


def build_workout_page(rng):
    workout = rng.choice(WORKOUT_TYPES)
    duration = rng.choice(WORKOUT_DURATIONS)
    calories = rng.choice(CALORIES_BURNED)
    hr = rng.choice(HEART_RATES)
    date = rng.choice(DATES)
    domain = rng.choice(DOMAINS)
    wid = f"w-{rng.randint(1000, 9999)}"

    body = (
        f'<div class="workout-summary" data-workout-id="{wid}">'
        f'<h1 class="workout-name">{workout}</h1>'
        f'<div class="workout-date"><time>{date}</time></div>'
        f'<div class="workout-stats">'
        f'<div class="stat"><span class="stat-label">Duration</span><span id="workout-duration" class="stat-value">{duration}</span></div>'
        f'<div class="stat"><span class="stat-label">Calories</span><span id="calories-burned" class="stat-value">{calories}</span></div>'
        f'<div class="stat"><span class="stat-label">Avg Heart Rate</span><span id="avg-hr" class="stat-value">{hr}</span></div></div>'
        f'<div class="exercise-list"><h2>Exercises</h2>'
        f'<div class="exercise"><span class="exercise-name">Bench Press</span><span class="sets">3 x 10</span></div>'
        f'<div class="exercise"><span class="exercise-name">Pull-ups</span><span class="sets">3 x 8</span></div></div></div>'
    )
    html = _wrap_html(f"{workout} - Fitness Tracker", body, rng=rng,
                       nav_items=[("Dashboard", "/"), ("Workouts", "/workouts"), ("Progress", "/progress")])

    targets = [
        ("#workout-duration", duration),
        ("#calories-burned", calories),
        ("#avg-hr", hr),
        (".workout-name", workout),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://fitness.{domain}/workouts/{wid}"
    return url, html, selector, expected


def build_wiki_page(rng):
    title = rng.choice(WIKI_TITLES)
    updated_by = rng.choice(WIKI_UPDATED_BY)
    date = rng.choice(DATES)
    domain = rng.choice(DOMAINS)
    slug = title.lower().replace(" ", "-").replace("'", "")[:30]

    toc_items = rng.sample(["Overview", "Prerequisites", "Installation", "Configuration",
                            "Usage", "Examples", "Troubleshooting", "FAQ", "Reference"], k=rng.randint(3, 5))
    toc = "".join(f'<li><a href="#{t.lower()}">{t}</a></li>' for t in toc_items)

    body = (
        f'<div class="wiki-page">'
        f'<div class="wiki-breadcrumb"><a href="/docs">Docs</a> / <a href="/docs/guides">Guides</a> / {title}</div>'
        f'<h1 class="wiki-title">{title}</h1>'
        f'<div class="wiki-meta"><span id="last-updated">Last updated: {date}</span>'
        f'<span id="updated-by">{updated_by}</span></div>'
        f'<div class="table-of-contents"><h3>Contents</h3><ol>{toc}</ol></div>'
        f'<div class="wiki-content"><h2>{toc_items[0]}</h2>'
        f'<p>This guide covers {title.lower()} in detail. Follow the steps below to get started.</p>'
        f'<h2>{toc_items[1] if len(toc_items) > 1 else "Details"}</h2>'
        f'<p>Make sure you have the required dependencies installed before proceeding.</p>'
        f'<pre><code>pip install example-package>=2.0</code></pre></div></div>'
    )
    html = _wrap_html(f"{title} - Documentation", body, rng=rng,
                       nav_items=[("Docs", "/docs"), ("Guides", "/guides"), ("API", "/api")])

    targets = [
        ("#last-updated", f"Last updated: {date}"),
        ("#updated-by", updated_by),
        (".wiki-title", title),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://docs.{domain}/wiki/{slug}"
    return url, html, selector, expected


def build_survey_page(rng):
    question = rng.choice(SURVEY_QUESTIONS)
    score = rng.choice(SURVEY_SCORES)
    responses = rng.choice(RESPONSE_COUNTS)
    domain = rng.choice(DOMAINS)
    survey_id = f"srv-{rng.randint(100, 999)}"

    body = (
        f'<div class="survey-results">'
        f'<h1>Survey Results</h1>'
        f'<div class="survey-question"><h2 class="question-text">{question}</h2></div>'
        f'<div class="survey-stats">'
        f'<span id="avg-score">Average Score: {score}</span>'
        f'<span id="response-count">{responses}</span></div>'
        f'<div class="score-distribution">'
        f'<div class="bar" data-score="5" style="width:40%"><span>5 stars: 40%</span></div>'
        f'<div class="bar" data-score="4" style="width:30%"><span>4 stars: 30%</span></div>'
        f'<div class="bar" data-score="3" style="width:15%"><span>3 stars: 15%</span></div></div></div>'
    )
    html = _wrap_html("Survey Results", body, rng=rng)

    targets = [
        ("#avg-score", f"Average Score: {score}"),
        ("#response-count", responses),
        (".question-text", question),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/surveys/{survey_id}/results"
    return url, html, selector, expected


def build_inventory_page(rng):
    item_name, sku, quantity = rng.choice(INVENTORY_ITEMS)
    warehouse = rng.choice(WAREHOUSE_LOCATIONS)
    domain = rng.choice(DOMAINS)
    last_restock = rng.choice(DATES)

    other_items = rng.sample([it for it in INVENTORY_ITEMS if it[1] != sku], k=rng.randint(2, 4))
    rows = "".join(
        f'<tr><td class="item-name">{n}</td><td class="sku">{s}</td>'
        f'<td class="qty">{q}</td><td class="loc">{rng.choice(WAREHOUSE_LOCATIONS)}</td></tr>'
        for n, s, q in other_items
    )
    body = (
        f'<div class="inventory-page">'
        f'<h1>Inventory Management</h1>'
        f'<div class="item-detail" data-sku="{sku}">'
        f'<h2 class="product-name">{item_name}</h2>'
        f'<div class="item-info">'
        f'<span id="item-sku">SKU: {sku}</span>'
        f'<span id="stock-quantity">Quantity: {quantity}</span>'
        f'<span id="warehouse-location">Location: {warehouse}</span>'
        f'<span class="last-restock">Last restock: {last_restock}</span></div></div>'
        f'<table class="inventory-table"><thead><tr><th>Item</th><th>SKU</th><th>Qty</th><th>Location</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
    html = _wrap_html("Inventory - Warehouse", body, rng=rng,
                       nav_items=[("Dashboard", "/"), ("Inventory", "/inventory"), ("Orders", "/orders")])

    targets = [
        ("#stock-quantity", f"Quantity: {quantity}"),
        ("#warehouse-location", f"Location: {warehouse}"),
        ("#item-sku", f"SKU: {sku}"),
        (".product-name", item_name),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://warehouse.{domain}/inventory/{sku}"
    return url, html, selector, expected


def build_dns_page(rng):
    domain_managed = rng.choice(DOMAINS)
    rec_type = rng.choice(DNS_RECORD_TYPES)
    value = rng.choice(DNS_VALUES)
    ttl = rng.choice(TTL_VALUES)
    host_domain = rng.choice([d for d in DOMAINS if d != domain_managed])

    other_records = [(rng.choice(DNS_RECORD_TYPES), rng.choice(DNS_VALUES), rng.choice(TTL_VALUES))
                     for _ in range(rng.randint(2, 4))]
    rows = "".join(
        f'<tr><td class="rec-type">{rt}</td><td class="rec-value">{rv}</td><td class="rec-ttl">{rttl}s</td></tr>'
        for rt, rv, rttl in other_records
    )
    body = (
        f'<div class="dns-manager">'
        f'<h1>DNS Records for {domain_managed}</h1>'
        f'<div class="primary-record">'
        f'<span id="record-type">Type: {rec_type}</span>'
        f'<span id="record-value">Value: {value}</span>'
        f'<span id="record-ttl">TTL: {ttl}s</span></div>'
        f'<table class="dns-table"><thead><tr><th>Type</th><th>Value</th><th>TTL</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
    html = _wrap_html(f"DNS - {domain_managed}", body, rng=rng)

    targets = [
        ("#record-type", f"Type: {rec_type}"),
        ("#record-value", f"Value: {value}"),
        ("#record-ttl", f"TTL: {ttl}s"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://dns.{host_domain}/zones/{domain_managed}"
    return url, html, selector, expected


def build_blog_post_page(rng):
    title = rng.choice(BLOG_TITLES)
    author = rng.choice(PERSON_NAMES)
    date = rng.choice(DATES)
    tags = rng.sample(BLOG_TAGS, k=rng.randint(2, 4))
    domain = rng.choice(DOMAINS)

    comments = []
    for _ in range(rng.randint(2, 5)):
        commenter = rng.choice(PERSON_NAMES)
        cbody = rng.choice(COMMENT_BODIES)
        comments.append(f'<div class="comment"><strong class="comment-author">{commenter}</strong><p>{cbody}</p></div>')

    tags_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    body = (
        f'<article class="blog-post">'
        f'<h1 class="post-title">{title}</h1>'
        f'<div class="post-meta"><span id="post-author">By {author}</span>'
        f'<time id="post-date">{date}</time></div>'
        f'<div class="post-tags">{tags_html}</div>'
        f'<div class="post-body"><p>This is a detailed technical post about modern engineering practices...</p></div>'
        f'<section class="comments"><h2>Comments ({len(comments)})</h2>'
        f'{"".join(comments)}</section></article>'
    )
    html = _wrap_html(f"{title} - Blog", body, rng=rng)
    slug = title.lower().replace(" ", "-").replace("'", "")[:40]

    targets = [
        (".post-title", title),
        ("#post-author", f"By {author}"),
        ("#post-date", date),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/blog/{slug}"
    return url, html, selector, expected


def build_pricing_page(rng):
    company = rng.choice(COMPANY_NAMES)
    domain = rng.choice(DOMAINS)
    highlighted = rng.choice(PLAN_TIERS)

    cards = []
    for tier in PLAN_TIERS:
        hl = ' data-highlighted="true"' if tier["name"] == highlighted["name"] else ""
        cards.append(
            f'<div class="pricing-card"{hl}>'
            f'<h3 class="tier-name">{tier["name"]}</h3>'
            f'<div class="tier-price">{tier["price"]}</div>'
            f'<p class="tier-features">{tier["features"]}</p>'
            f'<button>Choose Plan</button></div>'
        )
    body = (
        f'<div class="pricing-page">'
        f'<h1>{company} Pricing</h1>'
        f'<p class="pricing-tagline">Choose the plan that fits your needs</p>'
        f'<div class="pricing-grid">{"".join(cards)}</div>'
        f'<p id="highlighted-plan">Most popular: {highlighted["name"]} at {highlighted["price"]}</p>'
        f'</div>'
    )
    html = _wrap_html(f"Pricing - {company}", body, rng=rng)

    targets = [
        ("#highlighted-plan", f"Most popular: {highlighted['name']} at {highlighted['price']}"),
        (".pricing-tagline", "Choose the plan that fits your needs"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/pricing"
    return url, html, selector, expected


def build_file_manager_page(rng):
    domain = rng.choice(DOMAINS)
    files = rng.sample(FILE_NAMES, k=rng.randint(3, 6))
    target_file = rng.choice(files)
    size = rng.choice(FILE_SIZES)
    modified = rng.choice(FILE_MODIFIED_DATES)
    owner = rng.choice(PERSON_NAMES)

    rows = []
    for f in files:
        s = rng.choice(FILE_SIZES)
        m = rng.choice(FILE_MODIFIED_DATES)
        if f == target_file:
            s = size
            m = modified
        rows.append(
            f'<tr class="file-row" data-name="{f}">'
            f'<td class="file-name">{f}</td><td class="file-size">{s}</td>'
            f'<td class="file-modified">{m}</td><td class="file-owner">{owner}</td></tr>'
        )
    body = (
        f'<div class="file-manager">'
        f'<h1>Files</h1>'
        f'<div id="current-path">/shared/documents</div>'
        f'<table class="file-list"><thead><tr><th>Name</th><th>Size</th><th>Modified</th><th>Owner</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
        f'<div id="file-count">{len(files)} files</div></div>'
    )
    html = _wrap_html(f"File Manager - {domain}", body, rng=rng)

    targets = [
        (f'[data-name="{target_file}"] .file-size', size),
        (f'[data-name="{target_file}"] .file-modified', modified),
        ("#file-count", f"{len(files)} files"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/files/shared/documents"
    return url, html, selector, expected


def build_chat_page(rng):
    domain = rng.choice(DOMAINS)
    agent_name = rng.choice(PERSON_NAMES)
    customer = rng.choice(PERSON_NAMES)
    ticket_id = rng.choice(TICKET_IDS)

    msgs = []
    for i in range(rng.randint(2, 4)):
        cmsg = rng.choice(CHAT_MESSAGES)
        resp = rng.choice(CHAT_RESPONSES)
        msgs.append(f'<div class="msg customer"><span class="sender">{customer}</span><p>{cmsg}</p></div>')
        msgs.append(f'<div class="msg agent"><span class="sender">{agent_name}</span><p>{resp}</p></div>')

    last_response = rng.choice(CHAT_RESPONSES)
    body = (
        f'<div class="chat-window">'
        f'<div class="chat-header"><span id="chat-ticket">{ticket_id}</span>'
        f'<span id="chat-agent">Agent: {agent_name}</span></div>'
        f'<div class="chat-messages">{"".join(msgs)}'
        f'<div class="msg agent latest"><span class="sender">{agent_name}</span>'
        f'<p id="last-response">{last_response}</p></div></div></div>'
    )
    html = _wrap_html(f"Support Chat - {ticket_id}", body, rng=rng)

    targets = [
        ("#chat-agent", f"Agent: {agent_name}"),
        ("#chat-ticket", ticket_id),
        ("#last-response", last_response),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://support.{domain}/chat/{ticket_id}"
    return url, html, selector, expected


def build_audit_log_page(rng):
    domain = rng.choice(DOMAINS)
    entries = []
    for _ in range(rng.randint(4, 8)):
        action = rng.choice(AUDIT_ACTIONS)
        user = rng.choice(PERSON_NAMES)
        ip = rng.choice(AUDIT_IPS)
        date = rng.choice(DATES)
        entries.append(
            f'<tr class="audit-entry">'
            f'<td class="audit-action">{action}</td><td class="audit-user">{user}</td>'
            f'<td class="audit-ip">{ip}</td><td class="audit-date">{date}</td></tr>'
        )

    latest_action = rng.choice(AUDIT_ACTIONS)
    latest_user = rng.choice(PERSON_NAMES)
    latest_ip = rng.choice(AUDIT_IPS)
    body = (
        f'<div class="audit-log">'
        f'<h1>Audit Log</h1>'
        f'<div class="latest-entry">'
        f'<span id="latest-action">{latest_action}</span>'
        f'<span id="latest-user">{latest_user}</span>'
        f'<span id="latest-ip">{latest_ip}</span></div>'
        f'<table class="audit-table"><thead><tr><th>Action</th><th>User</th><th>IP</th><th>Date</th></tr></thead>'
        f'<tbody>{"".join(entries)}</tbody></table>'
        f'<div id="total-entries">{len(entries) + 1} entries</div></div>'
    )
    html = _wrap_html(f"Audit Log - {domain}", body, rng=rng)

    targets = [
        ("#latest-action", latest_action),
        ("#latest-user", latest_user),
        ("#latest-ip", latest_ip),
        ("#total-entries", f"{len(entries) + 1} entries"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://admin.{domain}/audit-log"
    return url, html, selector, expected


def build_server_metrics_page(rng):
    server = rng.choice(SERVER_NAMES)
    cpu = rng.choice(CPU_VALUES)
    memory = rng.choice(MEMORY_VALUES)
    disk = rng.choice(DISK_VALUES)
    uptime = rng.choice(UPTIME_VALUES)
    domain = rng.choice(DOMAINS)

    other_servers = rng.sample([s for s in SERVER_NAMES if s != server], k=rng.randint(2, 4))
    rows = "".join(
        f'<tr><td>{s}</td><td>{rng.choice(CPU_VALUES)}</td>'
        f'<td>{rng.choice(MEMORY_VALUES)}</td><td>{rng.choice(DISK_VALUES)}</td></tr>'
        for s in other_servers
    )
    body = (
        f'<div class="server-metrics">'
        f'<h1>Server Monitoring</h1>'
        f'<div class="primary-server" data-server="{server}">'
        f'<h2 id="server-name">{server}</h2>'
        f'<span id="metric-cpu">CPU: {cpu}</span>'
        f'<span id="metric-memory">Memory: {memory}</span>'
        f'<span id="metric-disk">Disk: {disk}</span>'
        f'<span id="metric-uptime">Uptime: {uptime}</span></div>'
        f'<table class="server-table"><thead><tr><th>Server</th><th>CPU</th><th>Memory</th><th>Disk</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
    html = _wrap_html(f"Monitoring - {server}", body, rng=rng)

    targets = [
        ("#server-name", server),
        ("#metric-cpu", f"CPU: {cpu}"),
        ("#metric-memory", f"Memory: {memory}"),
        ("#metric-disk", f"Disk: {disk}"),
        ("#metric-uptime", f"Uptime: {uptime}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://monitoring.{domain}/servers/{server}"
    return url, html, selector, expected


def build_ci_pipeline_page(rng):
    pipeline = rng.choice(PIPELINE_NAMES)
    status = rng.choice(PIPELINE_STATUSES)
    duration = rng.choice(PIPELINE_DURATIONS)
    commit = rng.choice(COMMIT_SHAS)
    branch = rng.choice(BRANCH_NAMES)
    author = rng.choice(PERSON_NAMES)
    domain = rng.choice(DOMAINS)

    steps = []
    step_names = ["checkout", "install", "lint", "test", "build", "deploy"]
    for s in step_names[:rng.randint(3, 6)]:
        st = rng.choice(["passed", "failed", "running", "pending"])
        steps.append(f'<div class="pipeline-step" data-step="{s}"><span class="step-name">{s}</span><span class="step-status">{st}</span></div>')

    body = (
        f'<div class="ci-pipeline">'
        f'<h1>Pipeline: {pipeline}</h1>'
        f'<div class="pipeline-info">'
        f'<span id="pipeline-status">Status: {status}</span>'
        f'<span id="pipeline-duration">Duration: {duration}</span>'
        f'<span id="pipeline-commit">Commit: {commit}</span>'
        f'<span id="pipeline-branch">Branch: {branch}</span>'
        f'<span id="pipeline-author">Triggered by: {author}</span></div>'
        f'<div class="pipeline-steps">{"".join(steps)}</div></div>'
    )
    html = _wrap_html(f"CI/CD - {pipeline}", body, rng=rng)

    targets = [
        ("#pipeline-status", f"Status: {status}"),
        ("#pipeline-duration", f"Duration: {duration}"),
        ("#pipeline-commit", f"Commit: {commit}"),
        ("#pipeline-branch", f"Branch: {branch}"),
        ("#pipeline-author", f"Triggered by: {author}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://ci.{domain}/pipelines/{pipeline}/{commit}"
    return url, html, selector, expected


def build_stock_page(rng):
    idx = rng.randint(0, len(STOCK_SYMBOLS) - 1)
    symbol = STOCK_SYMBOLS[idx]
    price = rng.choice(STOCK_PRICES)
    change = rng.choice(STOCK_CHANGES)
    cap = rng.choice(MARKET_CAPS)
    volume = rng.choice(VOLUME_VALUES)
    domain = rng.choice(DOMAINS)

    other_stocks = rng.sample([s for s in STOCK_SYMBOLS if s != symbol], k=rng.randint(2, 4))
    rows = "".join(
        f'<tr><td>{s}</td><td>{rng.choice(STOCK_PRICES)}</td><td>{rng.choice(STOCK_CHANGES)}</td></tr>'
        for s in other_stocks
    )
    body = (
        f'<div class="stock-page">'
        f'<h1 class="stock-symbol">{symbol}</h1>'
        f'<div class="stock-info">'
        f'<span id="stock-price">Price: {price}</span>'
        f'<span id="stock-change">Change: {change}</span>'
        f'<span id="market-cap">Market Cap: {cap}</span>'
        f'<span id="stock-volume">Volume: {volume}</span></div>'
        f'<table class="watchlist"><thead><tr><th>Symbol</th><th>Price</th><th>Change</th></tr></thead>'
        f'<tbody>{rows}</tbody></table></div>'
    )
    html = _wrap_html(f"{symbol} Stock Quote", body, rng=rng)

    targets = [
        ("#stock-price", f"Price: {price}"),
        ("#stock-change", f"Change: {change}"),
        ("#market-cap", f"Market Cap: {cap}"),
        ("#stock-volume", f"Volume: {volume}"),
        (".stock-symbol", symbol),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://finance.{domain}/quote/{symbol}"
    return url, html, selector, expected


def build_booking_page(rng):
    hotel = rng.choice(HOTEL_NAMES)
    room = rng.choice(ROOM_TYPES)
    rate = rng.choice(ROOM_RATES)
    bstatus = rng.choice(BOOKING_STATUSES)
    bid = rng.choice(BOOKING_IDS)
    guest = rng.choice(PERSON_NAMES)
    domain = rng.choice(DOMAINS)
    check_in = rng.choice(DATES[:6])
    check_out = rng.choice(DATES[6:])

    body = (
        f'<div class="booking-page">'
        f'<h1>{hotel}</h1>'
        f'<div class="booking-details">'
        f'<span id="booking-id">Booking: {bid}</span>'
        f'<span id="booking-status">Status: {bstatus}</span>'
        f'<span id="room-type">Room: {room}</span>'
        f'<span id="room-rate">Rate: {rate}</span>'
        f'<span id="guest-name">Guest: {guest}</span>'
        f'<span id="check-in">Check-in: {check_in}</span>'
        f'<span id="check-out">Check-out: {check_out}</span></div></div>'
    )
    html = _wrap_html(f"Booking - {hotel}", body, rng=rng)

    targets = [
        ("#booking-id", f"Booking: {bid}"),
        ("#booking-status", f"Status: {bstatus}"),
        ("#room-type", f"Room: {room}"),
        ("#room-rate", f"Rate: {rate}"),
        ("#guest-name", f"Guest: {guest}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://reservations.{domain}/bookings/{bid}"
    return url, html, selector, expected


def build_review_page(rng):
    product = rng.choice(PRODUCT_NAMES)
    domain = rng.choice(DOMAINS)

    reviews = []
    for _ in range(rng.randint(3, 6)):
        reviewer = rng.choice(PERSON_NAMES)
        stars = rng.choice(REVIEW_STARS)
        rbody = rng.choice(REVIEW_BODIES)
        helpful = rng.choice(REVIEW_HELPFULNESS)
        badge = rng.choice(VERIFIED_BADGES)
        reviews.append(
            f'<div class="review"><span class="reviewer">{reviewer}</span>'
            f'<span class="stars">{stars}</span><span class="badge">{badge}</span>'
            f'<p class="review-body">{rbody}</p><span class="helpful">{helpful}</span></div>'
        )

    avg_rating = rng.choice(RATINGS)
    total_reviews = rng.choice(REVIEW_COUNTS)
    body = (
        f'<div class="review-page">'
        f'<h1>Reviews for {product}</h1>'
        f'<div class="review-summary">'
        f'<span id="avg-rating">Average: {avg_rating}</span>'
        f'<span id="total-reviews">{total_reviews}</span></div>'
        f'<div class="review-list">{"".join(reviews)}</div></div>'
    )
    html = _wrap_html(f"Reviews - {product}", body, rng=rng)

    targets = [
        ("#avg-rating", f"Average: {avg_rating}"),
        ("#total-reviews", total_reviews),
    ]
    selector, expected = rng.choice(targets)
    slug = product.lower().replace(" ", "-")
    url = f"https://{domain}/products/{slug}/reviews"
    return url, html, selector, expected


def build_settings_page(rng):
    domain = rng.choice(DOMAINS)
    user = rng.choice(PERSON_NAMES)
    cat_name, fields = rng.choice(SETTING_CATEGORIES)

    settings_html = []
    target_field = rng.choice(fields)
    target_value = rng.choice(SETTING_VALUES)
    for field in fields:
        val = target_value if field == target_field else rng.choice(SETTING_VALUES)
        fid = field.lower().replace(" ", "-").replace("(", "").replace(")", "")
        settings_html.append(
            f'<div class="setting-row"><label>{field}</label>'
            f'<span class="setting-value" id="setting-{fid}">{val}</span></div>'
        )

    body = (
        f'<div class="settings-page">'
        f'<h1>Settings</h1>'
        f'<div id="settings-user">Logged in as: {user}</div>'
        f'<h2 class="settings-category">{cat_name}</h2>'
        f'<div class="settings-group">{"".join(settings_html)}</div></div>'
    )
    html = _wrap_html(f"Settings - {domain}", body, rng=rng)

    fid = target_field.lower().replace(" ", "-").replace("(", "").replace(")", "")
    targets = [
        (f"#setting-{fid}", target_value),
        ("#settings-user", f"Logged in as: {user}"),
        (".settings-category", cat_name),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/settings/{cat_name.lower()}"
    return url, html, selector, expected


def build_contact_page(rng):
    company = rng.choice(COMPANY_NAMES)
    domain = rng.choice(DOMAINS)
    phone = f"+1 ({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"
    email = f"contact@{domain}"
    address = f"{rng.randint(100,9999)} {rng.choice(['Market', 'Mission', 'Broadway', 'Main', 'Oak', 'Pine'])} St, {rng.choice(CITIES)}"
    hours = rng.choice(["Mon-Fri 9AM-5PM PST", "Mon-Fri 8AM-6PM EST", "24/7 Support Available"])

    body = (
        f'<div class="contact-page">'
        f'<h1>Contact {company}</h1>'
        f'<div class="contact-info">'
        f'<p id="contact-phone">Phone: {phone}</p>'
        f'<p id="contact-email">Email: {email}</p>'
        f'<p id="contact-address">Address: {address}</p>'
        f'<p id="contact-hours">Hours: {hours}</p></div>'
        f'<form class="contact-form"><input name="name" placeholder="Your Name">'
        f'<textarea name="message" placeholder="Your message..."></textarea>'
        f'<button type="submit">Send Message</button></form></div>'
    )
    html = _wrap_html(f"Contact - {company}", body, rng=rng)

    targets = [
        ("#contact-phone", f"Phone: {phone}"),
        ("#contact-email", f"Email: {email}"),
        ("#contact-hours", f"Hours: {hours}"),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://{domain}/contact"
    return url, html, selector, expected


def build_event_detail_page(rng):
    etype = rng.choice(EVENT_TYPES)
    ename = rng.choice(EVENT_NAMES)
    venue = rng.choice(EVENT_VENUES)
    date = rng.choice(DATES)
    time_slot = rng.choice(EVENT_TIMES)
    speaker = rng.choice(EVENT_SPEAKERS)
    domain = rng.choice(DOMAINS)
    attendees = f"{rng.randint(50, 2000)} registered"
    price = rng.choice(["Free", "$25", "$49", "$99", "$199", "$299"])

    body = (
        f'<div class="event-detail">'
        f'<h1 class="event-name">{ename}</h1>'
        f'<span class="event-type-badge">{etype}</span>'
        f'<div class="event-info">'
        f'<p id="event-date">Date: {date}</p>'
        f'<p id="event-time">Time: {time_slot}</p>'
        f'<p id="event-venue">Venue: {venue}</p>'
        f'<p id="event-speaker">Speaker: {speaker}</p>'
        f'<p id="event-attendees">Attendees: {attendees}</p>'
        f'<p id="event-price">Price: {price}</p></div>'
        f'<button class="register-btn">Register Now</button></div>'
    )
    html = _wrap_html(f"{ename} - {etype}", body, rng=rng)

    targets = [
        ("#event-date", f"Date: {date}"),
        ("#event-venue", f"Venue: {venue}"),
        ("#event-speaker", f"Speaker: {speaker}"),
        ("#event-attendees", f"Attendees: {attendees}"),
        ("#event-price", f"Price: {price}"),
        (".event-name", ename),
    ]
    selector, expected = rng.choice(targets)
    slug = ename.lower().replace(" ", "-").replace(":", "")[:40]
    url = f"https://events.{domain}/{slug}"
    return url, html, selector, expected


def build_error_log_page(rng):
    domain = rng.choice(DOMAINS)
    service = rng.choice(["auth-service", "api-gateway", "payment-service", "user-service",
                           "notification-service", "search-service", "billing-service"])
    latest_error = rng.choice(ERROR_MESSAGES)
    latest_level = rng.choice(ERROR_LEVELS)
    latest_time = rng.choice(DATES)
    seen_count = rng.choice(ERROR_STACK_COUNTS)

    entries = []
    for _ in range(rng.randint(3, 6)):
        entries.append(
            f'<tr class="log-entry"><td class="log-level">{rng.choice(ERROR_LEVELS)}</td>'
            f'<td class="log-message">{rng.choice(ERROR_MESSAGES)}</td>'
            f'<td class="log-count">{rng.choice(ERROR_STACK_COUNTS)}</td></tr>'
        )

    body = (
        f'<div class="error-log">'
        f'<h1>Error Log: {service}</h1>'
        f'<div class="latest-error">'
        f'<span id="error-level">{latest_level}</span>'
        f'<span id="error-message">{latest_error}</span>'
        f'<span id="error-count">{seen_count}</span>'
        f'<span id="error-time">{latest_time}</span></div>'
        f'<table class="error-table"><thead><tr><th>Level</th><th>Message</th><th>Count</th></tr></thead>'
        f'<tbody>{"".join(entries)}</tbody></table></div>'
    )
    html = _wrap_html(f"Errors - {service}", body, rng=rng)

    targets = [
        ("#error-level", latest_level),
        ("#error-message", latest_error),
        ("#error-count", seen_count),
        ("#error-time", latest_time),
    ]
    selector, expected = rng.choice(targets)
    url = f"https://logs.{domain}/errors/{service}"
    return url, html, selector, expected


# All builder functions (48 categories)
PAGE_BUILDERS = [
    build_product_page, build_article_page, build_dashboard_page, build_status_page,
    build_incident_page, build_profile_page, build_ticket_page, build_weather_page,
    build_job_page, build_billing_page, build_flight_page, build_banking_page,
    build_course_page, build_tracking_page, build_movie_page, build_restaurant_page,
    build_podcast_page, build_leaderboard_page, build_realty_page, build_email_page,
    build_thermostat_page,
    build_forum_page, build_changelog_page, build_search_results_page,
    build_notification_page, build_calendar_page, build_cart_page,
    build_api_docs_page, build_recipe_page, build_workout_page,
    build_wiki_page, build_survey_page, build_inventory_page, build_dns_page,
    build_blog_post_page, build_pricing_page, build_file_manager_page,
    build_chat_page, build_audit_log_page, build_server_metrics_page,
    build_ci_pipeline_page, build_stock_page, build_booking_page,
    build_review_page, build_settings_page, build_contact_page,
    build_event_detail_page, build_error_log_page,
]


# ============================================================================
# Phrasing pools
# ============================================================================

THOUGHTS_FETCH = [
    "I need to fetch the page from the URL first.",
    "Let me start by retrieving the HTML content from the given URL.",
    "First, I'll download the web page.",
    "My first step is to fetch the raw HTML from this URL.",
    "I should begin by making a request to the URL to get the page content.",
    "Let me fetch the web page so I can work with its content.",
    "I'll start by pulling the HTML from the provided URL.",
    "Time to retrieve the page. I'll use fetch_url to get the HTML.",
]

THOUGHTS_SAVE = [
    "Now I'll save the fetched HTML to the specified path.",
    "Let me write the fetched content to the given file path.",
    "I should persist the HTML to disk before parsing it.",
    "Next, I'll save this HTML content to the local path.",
    "Time to save the page content to the filesystem.",
    "I'll store the HTML locally so I can extract content from it.",
    "Let me save this to the given path for extraction.",
]

THOUGHTS_EXTRACT = [
    "Now I'll extract the text content using the CSS selector.",
    "Let me pull the relevant text from the DOM at the specified location.",
    "Time to extract the content at the given selector.",
    "I'll use the selector to find and extract the target text.",
    "Next, I need to extract the text from the element matching the selector.",
    "Let me query the saved HTML using the CSS selector to get the text.",
    "I'll extract the content from the matching DOM element now.",
]

THOUGHTS_VALIDATE = [
    "Now I'll validate that the extracted content matches the expected text.",
    "Let me check if the extracted text contains what we expect.",
    "Time to verify the content against the expected text.",
    "I'll run the validation to see if the expected text is present.",
    "Let me confirm the extracted content contains the expected text.",
    "I need to validate whether the expected text appears in what was extracted.",
]

THOUGHTS_SUCCESS = [
    "All steps completed successfully. The validation passed.",
    "The validation succeeded. The expected text was found.",
    "Everything checks out. The content matches as expected.",
    "Validation complete. The expected text is present in the extracted content.",
    "Task finished successfully. The content validation passed.",
    "The extracted content contains the expected text. All done.",
]

THOUGHTS_FAIL_VALIDATE = [
    "The validation failed. The extracted content does not contain the expected text.",
    "The content does not match. The expected text was not found in the extracted content.",
    "Validation did not pass. There is a mismatch between the extracted and expected text.",
    "The expected text is not present in what was extracted. This is a validation failure.",
]

THOUGHTS_FAIL_SELECTOR = [
    "The selector did not match any element on the page. No content was extracted.",
    "No element was found matching the given selector. The extraction returned nothing.",
    "The CSS selector did not find any matching elements in the HTML.",
    "The extraction came back empty. The selector does not match any element on this page.",
]

THOUGHTS_FAIL_FETCH = [
    "The fetch failed with an error. I cannot proceed with the remaining steps.",
    "There was an error fetching the URL. The page could not be retrieved.",
    "The request to the URL failed. I will report this error.",
    "I got an error when trying to fetch the page. The URL may be down or unreachable.",
]

THOUGHTS_PARTIAL_MATCH = [
    "The validation returned a match, but the extracted content seems ambiguous. The expected substring appears in a different context than intended.",
    "While the expected text is technically present as a substring, the full extracted content suggests a different meaning.",
    "The match may be misleading. The expected text appears within a larger phrase that changes its meaning.",
    "I should flag this: the expected text was found, but the surrounding context alters its meaning.",
]

THOUGHTS_MULTI_MATCH = [
    "The selector matched multiple elements and the extraction returned concatenated text from all matches. I need to note this.",
    "Multiple elements matched the selector. The extracted text combines content from several elements.",
    "The extraction returned text from more than one element since the selector was broad.",
    "The CSS selector was too broad and matched several elements. The result is a concatenation of all their text.",
]

FINAL_SUCCESS = [
    'Validation succeeded. The element at selector "{selector}" contains the expected text "{expected}".',
    'Task complete. The text "{expected}" was found at "{selector}" on the page.',
    'Success: the content at "{selector}" matches. Found: "{expected}".',
    'Confirmed: "{expected}" is present in the element matching "{selector}".',
    'The page at {url} contains "{expected}" at selector "{selector}". Validation passed.',
    'All checks passed. Selector "{selector}" returned content containing "{expected}".',
]

FINAL_FAIL_VALIDATE = [
    'Validation failed. The element at selector "{selector}" contains "{found}" which does not include the expected text "{expected}".',
    'The content at "{selector}" does not match. Expected "{expected}" but found "{found}".',
    'Failure: selector "{selector}" returned "{found}", which does not contain "{expected}".',
    'Mismatch: expected to find "{expected}" at "{selector}", but the actual content is "{found}".',
]

FINAL_FAIL_SELECTOR = [
    'Validation failed. The selector "{selector}" did not match any element on the page at {url}.',
    'No element found for selector "{selector}" on the page. Cannot validate.',
    'Failure: the CSS selector "{selector}" returned no results on the fetched page.',
    'Could not complete validation. No DOM element matches "{selector}" on this page.',
]

FINAL_FAIL_FETCH = [
    'Could not complete validation. Fetching {url} failed with: {error}.',
    'Task failed. The URL {url} returned an error: {error}. Cannot proceed.',
    'Failure: unable to fetch the page at {url}. Error: {error}.',
    'The page could not be retrieved. {url} returned: {error}.',
]

FINAL_PARTIAL = [
    'Warning: validation returned a match, but the result may be misleading. Selector "{selector}" returned "{found}" which contains "{expected}" as a substring, but the full context suggests a different meaning.',
    'Partial match warning. The text at "{selector}" is "{found}". While it contains "{expected}", the surrounding context changes the meaning.',
    'Caution: "{expected}" appears in "{found}" at selector "{selector}", but the match is likely a false positive given the full text.',
]

FINAL_MULTI = [
    'Note: the selector "{selector}" matched multiple elements. The combined extracted text is "{found}". The expected text "{expected}" was found within this combined result, but the match may not be from the intended element.',
    'The selector "{selector}" returned text from multiple elements: "{found}". While "{expected}" appears in the result, verify it came from the correct element.',
    'Multiple matches for "{selector}": combined text is "{found}". Expected "{expected}" is present but may come from the wrong element.',
]

USER_TEMPLATES = [
    """You have access to the following tools. Use them in order to complete the task.

{tools_text}

Task:
1. Fetch the web page at: {url}
2. Save the HTML to: {path}
3. Extract text at CSS selector: {selector}
4. Validate the extracted text contains: "{expected_text}"

Use Thought/Action/Action Input/Observation format. Give a Final Answer when done.""",

    """You have the following tools:

{tools_text}

Fetch {url}, save to {path}, extract content at "{selector}", and verify it contains "{expected_text}". Use Thought/Action/Action Input/Observation steps, then give a Final Answer.""",

    """Tools available:

{tools_text}

Your goal: verify that the page at {url} displays "{expected_text}" in the element matching CSS selector "{selector}". Save the page to {path} as evidence. Work step by step using Thought, Action, Action Input, and Observation. End with a Final Answer.""",

    """Below are your available tools:

{tools_text}

Please perform the following steps:
- Retrieve the HTML from {url}
- Save it to {path}
- Use selector "{selector}" to extract text
- Confirm the text includes "{expected_text}"

Format: Thought / Action / Action Input / Observation. Conclude with Final Answer.""",

    """{tools_text}

I need you to check whether the web page at {url} contains the text "{expected_text}" at the DOM location "{selector}". Fetch the page, save it to {path}, extract the content, and validate. Use the Thought/Action/Action Input/Observation format and finish with a Final Answer.""",

    """Here are your tools:

{tools_text}

Task: Navigate to {url} and confirm the element "{selector}" shows "{expected_text}". Save the page locally at {path} as part of the process. Use the ReAct format: Thought, Action, Action Input, Observation. Conclude with Final Answer.""",

    """{tools_text}

Objective: Verify content on a web page.
- URL: {url}
- CSS Selector: {selector}
- Expected text: "{expected_text}"
- Save location: {path}

Execute each tool step by step. Use Thought/Action/Action Input/Observation and end with a Final Answer summarizing the result.""",
]

FETCH_ERRORS = [
    "HTTP 404: Not Found", "HTTP 500: Internal Server Error", "HTTP 503: Service Unavailable",
    "ConnectionError: Connection refused", "ConnectionError: Connection timed out after 30s",
    "HTTP 403: Forbidden - Access Denied", "SSLError: Certificate verification failed",
    "HTTP 502: Bad Gateway", "ConnectionError: DNS resolution failed",
    "HTTP 429: Too Many Requests", "TimeoutError: Read timed out after 60s",
]

PARTIAL_MATCH_CASES = [
    {"expected": "In Stock", "found": "Out of Stock"},
    {"expected": "Passed", "found": "Failed - 3 tests not passed"},
    {"expected": "Active", "found": "Inactive since January"},
    {"expected": "Approved", "found": "Not Approved - pending review"},
    {"expected": "Online", "found": "Last Online: 3 days ago"},
    {"expected": "Enabled", "found": "Disabled (was Enabled until March 1)"},
    {"expected": "Operational", "found": "Non-Operational since 08:00 UTC"},
    {"expected": "Complete", "found": "Incomplete - 3 items remaining"},
    {"expected": "Valid", "found": "Invalid certificate detected"},
    {"expected": "Delivered", "found": "Not yet Delivered - in transit"},
]

WRONG_SELECTORS = [
    "#nonexistent-id", ".missing-class", "#typo-elemnt", ".wrng-selector",
    "#outdated-widget", ".removed-section", "#old-banner", ".deprecated-panel",
    "#content-v1", ".legacy-header", "#deleted-notice", ".no-such-element",
    "#stale-ref", ".archived-box", "#removed-card", ".v2-container",
]

WRONG_TEXTS = [
    "Page not available", "N/A", "Coming Soon", "Temporarily Unavailable",
    "Under Maintenance", "No data", "Error loading content", "Please refresh",
    "Contact support", "Feature deprecated", "Redirecting...", "Loading...",
    "Access denied", "Session expired", "Content removed", "Not found",
    "Service unavailable", "Rate limited", "Quota exceeded", "Try again later",
]


def pick(rng, lst):
    return rng.choice(lst)


# ============================================================================
# Trace builders
# ============================================================================

def build_success_trace(rng, url, path, html, selector, expected_text):
    html_obs = html if len(html) <= 600 else html[:597] + "..."
    return "\n".join([
        f"Thought: {pick(rng, THOUGHTS_FETCH)}",
        "Action: fetch_url",
        f'Action Input: {{"url": "{url}"}}',
        f"Observation: {html_obs}",
        f"Thought: {pick(rng, THOUGHTS_SAVE)}",
        "Action: save_page",
        f'Action Input: {{"content": {json.dumps(html)}, "path": "{path}"}}',
        "Observation: Saved successfully.",
        f"Thought: {pick(rng, THOUGHTS_EXTRACT)}",
        "Action: extract_dom_content",
        f'Action Input: {{"path": "{path}", "selector": {json.dumps(selector)}}}',
        f"Observation: {json.dumps(expected_text)}",
        f"Thought: {pick(rng, THOUGHTS_VALIDATE)}",
        "Action: validate_content",
        f'Action Input: {{"extracted_content": {json.dumps(expected_text)}, "expected_text": {json.dumps(expected_text)}}}',
        "Observation: Match.",
        f"Thought: {pick(rng, THOUGHTS_SUCCESS)}",
        "Final Answer: " + pick(rng, FINAL_SUCCESS).format(
            selector=selector, expected=expected_text, url=url),
    ])


def build_fail_validate_trace(rng, url, path, html, selector, expected_text):
    found = pick(rng, WRONG_TEXTS)
    html_obs = html if len(html) <= 600 else html[:597] + "..."
    return "\n".join([
        f"Thought: {pick(rng, THOUGHTS_FETCH)}", "Action: fetch_url",
        f'Action Input: {{"url": "{url}"}}', f"Observation: {html_obs}",
        f"Thought: {pick(rng, THOUGHTS_SAVE)}", "Action: save_page",
        f'Action Input: {{"content": {json.dumps(html)}, "path": "{path}"}}', "Observation: Saved successfully.",
        f"Thought: {pick(rng, THOUGHTS_EXTRACT)}", "Action: extract_dom_content",
        f'Action Input: {{"path": "{path}", "selector": {json.dumps(selector)}}}',
        f"Observation: {json.dumps(found)}",
        f"Thought: {pick(rng, THOUGHTS_VALIDATE)}", "Action: validate_content",
        f'Action Input: {{"extracted_content": {json.dumps(found)}, "expected_text": {json.dumps(expected_text)}}}',
        "Observation: No match.",
        f"Thought: {pick(rng, THOUGHTS_FAIL_VALIDATE)}",
        "Final Answer: " + pick(rng, FINAL_FAIL_VALIDATE).format(
            selector=selector, expected=expected_text, found=found, url=url),
    ])


def build_fail_selector_trace(rng, url, path, html, selector, expected_text):
    wrong_sel = pick(rng, WRONG_SELECTORS)
    html_obs = html if len(html) <= 600 else html[:597] + "..."
    return wrong_sel, "\n".join([
        f"Thought: {pick(rng, THOUGHTS_FETCH)}", "Action: fetch_url",
        f'Action Input: {{"url": "{url}"}}', f"Observation: {html_obs}",
        f"Thought: {pick(rng, THOUGHTS_SAVE)}", "Action: save_page",
        f'Action Input: {{"content": {json.dumps(html)}, "path": "{path}"}}', "Observation: Saved successfully.",
        f"Thought: {pick(rng, THOUGHTS_EXTRACT)}", "Action: extract_dom_content",
        f'Action Input: {{"path": "{path}", "selector": {json.dumps(wrong_sel)}}}',
        'Observation: ""',
        f"Thought: {pick(rng, THOUGHTS_FAIL_SELECTOR)}",
        "Final Answer: " + pick(rng, FINAL_FAIL_SELECTOR).format(
            selector=wrong_sel, expected=expected_text, url=url),
    ])


def build_fail_fetch_trace(rng, url, path, selector, expected_text):
    error = pick(rng, FETCH_ERRORS)
    return "\n".join([
        f"Thought: {pick(rng, THOUGHTS_FETCH)}", "Action: fetch_url",
        f'Action Input: {{"url": "{url}"}}', f"Observation: Error: {error}",
        f"Thought: {pick(rng, THOUGHTS_FAIL_FETCH)}",
        "Final Answer: " + pick(rng, FINAL_FAIL_FETCH).format(url=url, error=error),
    ])


def build_partial_match_trace(rng, url, path, html, selector, expected_text):
    case = pick(rng, PARTIAL_MATCH_CASES)
    html_obs = html if len(html) <= 600 else html[:597] + "..."
    return case["expected"], "\n".join([
        f"Thought: {pick(rng, THOUGHTS_FETCH)}", "Action: fetch_url",
        f'Action Input: {{"url": "{url}"}}', f"Observation: {html_obs}",
        f"Thought: {pick(rng, THOUGHTS_SAVE)}", "Action: save_page",
        f'Action Input: {{"content": {json.dumps(html)}, "path": "{path}"}}', "Observation: Saved successfully.",
        f"Thought: {pick(rng, THOUGHTS_EXTRACT)}", "Action: extract_dom_content",
        f'Action Input: {{"path": "{path}", "selector": {json.dumps(selector)}}}',
        f'Observation: {json.dumps(case["found"])}',
        f"Thought: {pick(rng, THOUGHTS_VALIDATE)}", "Action: validate_content",
        f'Action Input: {{"extracted_content": {json.dumps(case["found"])}, "expected_text": {json.dumps(case["expected"])}}}',
        "Observation: Match.",
        f"Thought: {pick(rng, THOUGHTS_PARTIAL_MATCH)}",
        "Final Answer: " + pick(rng, FINAL_PARTIAL).format(
            selector=selector, expected=case["expected"], found=case["found"], url=url),
    ])


def build_multi_match_trace(rng, url, path, html, selector, expected_text):
    combined = expected_text + " | " + pick(rng, WRONG_TEXTS) + " | " + pick(rng, WRONG_TEXTS)
    html_obs = html if len(html) <= 600 else html[:597] + "..."
    return "\n".join([
        f"Thought: {pick(rng, THOUGHTS_FETCH)}", "Action: fetch_url",
        f'Action Input: {{"url": "{url}"}}', f"Observation: {html_obs}",
        f"Thought: {pick(rng, THOUGHTS_SAVE)}", "Action: save_page",
        f'Action Input: {{"content": {json.dumps(html)}, "path": "{path}"}}', "Observation: Saved successfully.",
        f"Thought: {pick(rng, THOUGHTS_EXTRACT)}", "Action: extract_dom_content",
        f'Action Input: {{"path": "{path}", "selector": {json.dumps(selector)}}}',
        f'Observation: {json.dumps(combined)}',
        f"Thought: {pick(rng, THOUGHTS_VALIDATE)}", "Action: validate_content",
        f'Action Input: {{"extracted_content": {json.dumps(combined)}, "expected_text": {json.dumps(expected_text)}}}',
        "Observation: Match.",
        f"Thought: {pick(rng, THOUGHTS_MULTI_MATCH)}",
        "Final Answer: " + pick(rng, FINAL_MULTI).format(
            selector=selector, expected=expected_text, found=combined, url=url),
    ])


FAILURE_MODES = ["fetch_error", "selector_miss", "validate_fail", "partial_match", "multi_match"]


def make_example(rng, tools_text, idx, failure_mode=None):
    builder = pick(rng, PAGE_BUILDERS)
    url, html, selector, expected_text = builder(rng)
    path = f"{pick(rng, PATH_PREFIXES)}/{url.split('//')[1].replace('/', '_').replace('.', '_').replace('?', '_')}_{idx}.html"

    user_template = pick(rng, USER_TEMPLATES)

    if failure_mode == "fetch_error":
        user_msg = user_template.format(
            tools_text=tools_text, url=url, path=path,
            selector=selector, expected_text=expected_text)
        assistant_msg = build_fail_fetch_trace(rng, url, path, selector, expected_text)
    elif failure_mode == "selector_miss":
        wrong_sel, assistant_msg = build_fail_selector_trace(rng, url, path, html, selector, expected_text)
        user_msg = user_template.format(
            tools_text=tools_text, url=url, path=path,
            selector=wrong_sel, expected_text=expected_text)
    elif failure_mode == "validate_fail":
        user_msg = user_template.format(
            tools_text=tools_text, url=url, path=path,
            selector=selector, expected_text=expected_text)
        assistant_msg = build_fail_validate_trace(rng, url, path, html, selector, expected_text)
    elif failure_mode == "partial_match":
        pm_expected, assistant_msg = build_partial_match_trace(rng, url, path, html, selector, expected_text)
        user_msg = user_template.format(
            tools_text=tools_text, url=url, path=path,
            selector=selector, expected_text=pm_expected)
    elif failure_mode == "multi_match":
        user_msg = user_template.format(
            tools_text=tools_text, url=url, path=path,
            selector=selector, expected_text=expected_text)
        assistant_msg = build_multi_match_trace(rng, url, path, html, selector, expected_text)
    else:
        user_msg = user_template.format(
            tools_text=tools_text, url=url, path=path,
            selector=selector, expected_text=expected_text)
        assistant_msg = build_success_trace(rng, url, path, html, selector, expected_text)

    return {
        "conversations": [
            {"from": "user", "value": user_msg.strip()},
            {"from": "assistant", "value": assistant_msg.strip()},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Generate diverse training data for agentic loop.")
    parser.add_argument("--total", type=int, default=2500, help="Total examples (default 2500).")
    parser.add_argument("--eval-ratio", type=float, default=0.175, help="Eval set fraction (default ~140 out of 800).")
    parser.add_argument("--failure-ratio", type=float, default=0.22, help="Fraction of failure cases (default ~22%%).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR, help="Output directory.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tools_def = load_tool_definitions()
    tools_text = format_tools_for_prompt(tools_def)

    n_failures = int(args.total * args.failure_ratio)
    n_success = args.total - n_failures

    examples = []
    for i in range(n_success):
        examples.append(make_example(rng, tools_text, i, failure_mode=None))
    for i in range(n_failures):
        mode = FAILURE_MODES[i % len(FAILURE_MODES)]
        examples.append(make_example(rng, tools_text, n_success + i, failure_mode=mode))

    rng.shuffle(examples)

    n_eval = max(1, int(len(examples) * args.eval_ratio))
    n_train = len(examples) - n_eval
    train_examples = examples[:n_train]
    eval_examples = examples[n_train:]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.out_dir / "train.jsonl"
    eval_path = args.out_dir / "eval.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(eval_path, "w", encoding="utf-8") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {n_train} train examples to {train_path}")
    print(f"Wrote {n_eval} eval examples to {eval_path}")
    print(f"  Success: {n_success}, Failure: {n_failures} ({len(FAILURE_MODES)} modes)")
    print(f"  Page builder categories: {len(PAGE_BUILDERS)}")


if __name__ == "__main__":
    main()
