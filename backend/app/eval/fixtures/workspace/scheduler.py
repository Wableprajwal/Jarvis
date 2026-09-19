from datetime import datetime, timedelta


def next_run_time(last_run: datetime, interval_minutes: int) -> datetime:
    """Return the next scheduled run, `interval_minutes` after the last run."""
    if interval_minutes <= 0:
        raise ValueError("interval must be positive")
    return last_run + timedelta(minutes=interval_minutes)
