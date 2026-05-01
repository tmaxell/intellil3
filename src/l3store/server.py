from __future__ import annotations

import argparse
import logging
from pathlib import Path

from l3store.api.grpc_server import serve
from l3store.utils.config import L3Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l3-store")
    subparsers = parser.add_subparsers(dest="command")

    serve_parser = subparsers.add_parser("serve", help="Run the L3 gRPC server")
    serve_parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to L3 YAML config",
    )
    serve_parser.add_argument("--host", default="[::]", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=50051, help="Bind port")
    serve_parser.add_argument(
        "--max-workers",
        type=int,
        default=10,
        help="gRPC worker thread count",
    )
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "serve":
        parser.print_help()
        return

    logging.basicConfig(level=getattr(logging, args.log_level))
    config = L3Config.from_yaml(args.config)
    serve(
        config,
        host=args.host,
        port=args.port,
        max_workers=args.max_workers,
        wait=True,
    )


if __name__ == "__main__":
    main()
