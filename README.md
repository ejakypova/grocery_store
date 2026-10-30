# Grocery scraper -> PostgreSQL

1. Create a PostgreSQL database: `CREATE DATABASE grocery_ds;`
2. `pip install -r requirements.txt`
3. `python -m playwright install chromium`
4. Set `DATABASE_URL` in your terminal using the pattern in `.env.example` (the script reads environment variables; it does not automatically load .env).
5. Run `python parser.py --stores Globus Dostor Optobaza`

The scraper reuses experimental site selectors from the earlier version. Real website extraction has not been verified. Respect robots.txt, site terms and rate limits. The database schema is created automatically. Each run adds a price observation, even when the price is unchanged. No Excel export is generated.
