"""Lightweight training health check; does not load strategy tables."""
import argparse
import json
from pathlib import Path
import shutil
import time
from .checkpoints import save


def status(directory):
    root=Path(directory).resolve()
    workers=[]
    for worker in sorted(root.glob('seed-*')):
        if not worker.is_dir():
            continue
        checkpoint=worker/'latest.sqlite' if (worker/'latest.sqlite').exists() else worker/'latest.json.gz'
        metrics=worker/'metrics.jsonl'
        latest=None
        if metrics.exists():
            # Read line by line: ignore an incomplete in-progress write.
            for line in metrics.open():
                try:
                    latest=json.loads(line)
                except json.JSONDecodeError:
                    pass
        workers.append(dict(seed=worker.name,checkpoint_exists=checkpoint.exists(),
                            checkpoint_age_seconds=time.time()-checkpoint.stat().st_mtime if checkpoint.exists() else None,
                            last_saved=latest))
    processes=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if b'-m' not in args or not any(x in args for x in (b'cortado.parallel',b'cortado.run',b'cortado.campaign',b'cortado.evaluate_workers')):
                continue
            if not any(str(root).encode() in arg for arg in args):
                # Commands can have cwd-relative paths.
                cwd=(proc/'cwd').resolve()
                if not any((cwd/arg.decode()).resolve()==root for arg in args if arg):
                    continue
            info=(proc/'status').read_text()
            rss=next((int(line.split()[1])/1024 for line in info.splitlines() if line.startswith('VmRSS:')),0)
            processes.append(dict(pid=int(proc.name),rss_mb=rss,module=next((arg.decode() for arg in args if arg.startswith(b'cortado.')),'')))
        except (OSError,UnicodeError,ValueError):
            continue
    return dict(schema=1,checked_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                run_dir=str(root),processes=processes,workers=workers,
                free_disk_gib=shutil.disk_usage(root).free/1024**3,
                campaign=json.loads((root/'campaign.json').read_text()) if (root/'campaign.json').exists() else None,
                summary=json.loads((root/'experiment-summary.json').read_text()) if (root/'experiment-summary.json').exists() else None)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    args=p.parse_args()
    report=status(args.run_dir)
    save(Path(args.run_dir)/'health.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
