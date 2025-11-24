import argparse
import dataclasses


@dataclasses.dataclass
class LBArgs:
    rust_lb: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    policy: str = "random"
    encode_infos: list = dataclasses.field(default_factory=list)
    prefill_infos: list = dataclasses.field(default_factory=list)
    decode_infos: list = dataclasses.field(default_factory=list)
    text_infos: list = dataclasses.field(default_factory=list)
    log_interval: int = 5
    timeout: int = 600

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--rust-lb",
            action="store_true",
            help="Use Rust load balancer",
        )
        parser.add_argument(
            "--host",
            type=str,
            default=LBArgs.host,
            help=f"Host to bind the server (default: {LBArgs.host})",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=LBArgs.port,
            help=f"Port to bind the server (default: {LBArgs.port})",
        )
        parser.add_argument(
            "--policy",
            type=str,
            default=LBArgs.policy,
            choices=["random", "po2"],
            help=f"Policy to use for load balancing (default: {LBArgs.policy})",
        )
        parser.add_argument(
            "--encode",
            type=str,
            default=[],
            nargs="+",
            help="URLs for encode servers",
        )
        parser.add_argument(
            "--prefill",
            type=str,
            default=[],
            nargs="+",
            help="URLs for prefill servers",
        )
        parser.add_argument(
            "--decode",
            type=str,
            default=[],
            nargs="+",
            help="URLs for decode servers",
        )
        parser.add_argument(
            "--text",
            type=str,
            default=[],
            nargs="+",
            help="URLs for non-disaggregated text servers. This only makes sense when encoder is disaggregated",
        )
        parser.add_argument(
            "--prefill-bootstrap-ports",
            type=int,
            nargs="+",
            help="Bootstrap ports for prefill servers",
        )
        parser.add_argument(
            "--encode-bootstrap-ports",
            type=int,
            nargs="+",
            help="Bootstrap ports for encode servers",
        )

        parser.add_argument(
            "--log-interval",
            type=int,
            default=LBArgs.log_interval,
            help=f"Log interval in seconds (default: {LBArgs.log_interval})",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=LBArgs.timeout,
            help=f"Timeout in seconds (default: {LBArgs.timeout})",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "LBArgs":
        prefill_bootstrap_ports = args.prefill_bootstrap_ports
        if prefill_bootstrap_ports is None:
            prefill_bootstrap_ports = [None] * len(args.prefill)
        elif len(prefill_bootstrap_ports) == 1:
            prefill_bootstrap_ports = prefill_bootstrap_ports * len(args.prefill)
        else:
            if len(prefill_bootstrap_ports) != len(args.prefill):
                raise ValueError(
                    "Number of prefill URLs must match number of bootstrap ports"
                )

        encode_bootstrap_ports = args.encode_bootstrap_ports
        if encode_bootstrap_ports is None:
            encode_bootstrap_ports = [None] * len(args.encode)
        elif len(encode_bootstrap_ports) == 1:
            encode_bootstrap_ports = encode_bootstrap_ports * len(args.encode)
        else:
            if len(encode_bootstrap_ports) != len(args.encode):
                raise ValueError(
                    "Number of encode URLs must match number of bootstrap ports"
                )

        prefill_infos = [
            (url, port) for url, port in zip(args.prefill, prefill_bootstrap_ports)
        ]
        encode_infos = [
            (url, port) for url, port in zip(args.encode, encode_bootstrap_ports)
        ]

        if args.encode:
            assert not args.rust_lb, "encode disaggregation is not supported in rust lb"
            assert (
                args.prefill and args.decode
            ) or args.text, "Both p and d or p-and-d should be specified under encoder disaggregation"

        if args.text is not None:
            assert (
                args.encode is not None
            ), "Non-disaggregated pd must work with encoder disaggregated"

        return cls(
            rust_lb=args.rust_lb,
            host=args.host,
            port=args.port,
            policy=args.policy,
            encode_infos=encode_infos,
            prefill_infos=prefill_infos,
            decode_infos=args.decode,
            text_infos=args.text,
            log_interval=args.log_interval,
            timeout=args.timeout,
        )

    def __post_init__(self):
        if not self.rust_lb:
            assert (
                self.policy == "random"
            ), "Only random policy is supported for Python load balancer"


def main():
    parser = argparse.ArgumentParser(
        description="PD Disaggregation Load Balancer Server"
    )
    LBArgs.add_cli_args(parser)
    args = parser.parse_args()
    lb_args = LBArgs.from_cli_args(args)

    if lb_args.rust_lb:
        from sgl_pdlb._rust import LoadBalancer as RustLB

        RustLB(
            host=lb_args.host,
            port=lb_args.port,
            policy=lb_args.policy,
            prefill_infos=lb_args.prefill_infos,
            decode_infos=lb_args.decode_infos,
            log_interval=lb_args.log_interval,
            timeout=lb_args.timeout,
        ).start()
    else:
        from sglang.srt.disaggregation.mini_lb import PrefillConfig, run

        prefill_configs = [
            PrefillConfig(url, port) for url, port in lb_args.prefill_infos
        ]
        encode_configs = [
            PrefillConfig(url, port) for url, port in lb_args.encode_infos
        ]

        run(
            prefill_configs,
            lb_args.decode_infos,
            encode_configs,
            lb_args.text_infos,
            lb_args.host,
            lb_args.port,
        )


if __name__ == "__main__":
    main()
