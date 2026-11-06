export type CliArgs = {
  command: "chat" | "monitor";
  forceSetup: boolean;
  profile?: string;
  socket?: string;
  db?: string;
  artifact?: string;
  alloydBin?: string;
  message?: string;
  resume?: string;
  help: boolean;
  once: boolean;
  json: boolean;
  intervalMs: number;
  powermetrics: boolean;
};

export function parseArgv(argv: string[]): CliArgs {
  const args = argv.slice(2);
  let i = 0;
  const command = args[0] === "monitor" ? "monitor" : "chat";
  if (args[0] === "chat" || args[0] === "monitor") {
    i = 1;
  }

  const out: CliArgs = {
    command,
    forceSetup: false,
    help: false,
    once: false,
    json: false,
    intervalMs: 1000,
    powermetrics: false,
  };

  for (; i < args.length; i++) {
    const a = args[i]!;
    const need = (): string => {
      const value = args[++i];
      if (value === undefined) {
        throw new Error(`${a} requires a value`);
      }
      return value;
    };
    const monitorOnly = (): void => {
      if (command !== "monitor") {
        throw new Error(`${a} is only valid with alloy monitor`);
      }
    };

    switch (a) {
      case "-h":
      case "--help":
        out.help = true;
        break;
      case "--setup":
        out.forceSetup = true;
        break;
      case "--profile":
        out.profile = need();
        break;
      case "--socket":
        out.socket = need();
        break;
      case "--db":
        out.db = need();
        break;
      case "--artifact":
        out.artifact = need();
        break;
      case "--alloyd-bin":
        out.alloydBin = need();
        break;
      case "-m":
      case "--message":
        out.message = need();
        break;
      case "--resume":
        out.resume = need();
        break;
      case "--once":
        monitorOnly();
        out.once = true;
        break;
      case "--json":
        monitorOnly();
        out.json = true;
        break;
      case "--interval": {
        monitorOnly();
        const value = need();
        const interval = Number(value);
        if (!/^[0-9]+$/.test(value) || !Number.isSafeInteger(interval) || interval < 500 || interval > 60_000) {
          throw new Error("--interval must be an integer from 500 to 60000 milliseconds");
        }
        out.intervalMs = interval;
        break;
      }
      case "--powermetrics":
        monitorOnly();
        out.powermetrics = true;
        break;
      default:
        throw new Error(`unknown argument: ${a}`);
    }
  }

  if (command === "monitor") {
    if (out.forceSetup || out.message !== undefined || out.resume !== undefined ||
        out.profile !== undefined || out.db !== undefined || out.artifact !== undefined ||
        out.alloydBin !== undefined) {
      throw new Error("alloy monitor accepts --socket, --once, --json, --interval, and --powermetrics; chat/setup options do not apply");
    }
  }

  return out;
}
