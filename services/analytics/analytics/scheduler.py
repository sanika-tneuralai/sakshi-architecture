"""
Analytics scheduler for daily aggregation jobs.
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
from analytics.service import aggregate_daily_analytics


scheduler = BackgroundScheduler()


def daily_aggregation_job():
    """Cron job that runs daily to aggregate analytics."""
    print(f"\n{'='*60}")
    print(f"[CRON JOB] Daily Analytics Aggregation Started")
    print(f"[CRON JOB] Timestamp: {datetime.utcnow().isoformat()}")
    print(f"{'='*60}")
    
    try:
        aggregate_daily_analytics()
        print(f"[CRON JOB] Daily Analytics Aggregation Completed Successfully")
    except Exception as e:
        print(f"[CRON JOB] Daily Analytics Aggregation Failed: {str(e)}")
    
    print(f"{'='*60}\n")


def start_scheduler():
    """Start the analytics scheduler."""
    print(f"[SCHEDULER] Starting analytics scheduler")
    
    # Schedule daily aggregation at 00:30 UTC
    scheduler.add_job(
        daily_aggregation_job,
        trigger=CronTrigger(hour=0, minute=30, timezone='UTC'),
        id='daily_aggregation',
        name='Daily Analytics Aggregation',
        replace_existing=True
    )
    
    scheduler.start()
    print(f"[SCHEDULER] Scheduler started - Daily aggregation scheduled for 00:30 UTC")


def stop_scheduler():
    """Stop the analytics scheduler."""
    print(f"[SCHEDULER] Stopping analytics scheduler")
    scheduler.shutdown(wait=False)
    print(f"[SCHEDULER] Scheduler stopped")
