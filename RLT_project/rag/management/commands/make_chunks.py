from django.core.management.base import BaseCommand
from ...chunking import build_all_chunks, OUT_JSONL

class Command(BaseCommand):
    help = "Разбить parsed_data.json на тематические чанки с контекстом"

    def add_arguments(self, parser):
        parser.add_argument("--max-chars", type=int, default=1100)
        parser.add_argument("--overlap", type=int, default=2)

    def handle(self, *args, **opts):
        n = build_all_chunks(max_chars=opts["max_chars"], overlap_sentences=opts["overlap"])
        self.stdout.write(self.style.SUCCESS(
            f"Готово: {n} чанков → {OUT_JSONL}"
        ))