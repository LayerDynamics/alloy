export type CliArgs = {
  forceSetup: boolean;
  profile?: string;
  socket?: string;
  db?: string;
  artifact?: string;
  alloydBin?: string;
  message?: string;
  resume?: string;
  help: boolean;
};

export function parseArgv(argv: string[]): CliArgs {
  const args = argv.slice(2);
  let i = 0;
  if (args[0] === "chat") {
    i = 1;
  }

  const out: CliArgs = { forceSetup: false, help: false };

  for (; i < args.length; i++) {
    const a = args[i]!;
    const need = (): string => {
      const value = args[++i];
      if (value === undefined) {
        throw new Error(`${a} requires a value`);
      }
      return value;
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
      default:
        throw new Error(`unknown argument: ${a}`);
    }
  }

  return out;
}
