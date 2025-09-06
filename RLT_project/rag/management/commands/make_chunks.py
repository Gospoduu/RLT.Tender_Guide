from django.core.management.base import BaseCommand
from ...chunking import build_all_chunks

class Command(BaseCommand):
    help = "Чанкует parsed_data.json в тематические чанки"

    def handle(self, *args, **kwargs):
        count, path = build_all_chunks()
        self.stdout.write(self.style.SUCCESS(
            f"✅ Успешно: создано {count} чанков → {path}"
        ))