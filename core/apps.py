
from django.apps import AppConfig
from django.db.utils import ProgrammingError
import logging

logger = logging.getLogger(__name__)

class RecommenderApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        """
        This method is called when the app is ready. It's the ideal place
        to perform startup tasks, such as clearing a database table.
        """
        try:
            from .models import RecommendationLog, Order
            logger.info("Server restarted. Clearing all RecommendationLog and Order records.")
            # Delete all recommendation logs from the database
            RecommendationLog.objects.all().delete()
            # Delete all orders from the database
            Order.objects.all().delete()
            logger.info("Successfully cleared RecommendationLog and Order records.")
        except ProgrammingError:
            # This handles the case where the table doesn't exist yet (e.g., on a fresh migration)
            logger.warning("RecommendationLog or Order table does not exist yet. Skipping cleanup.")
        except Exception as e:
            logger.error(f"Failed to clear RecommendationLog or Order records: {e}")