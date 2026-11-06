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
export declare function parseArgv(argv: string[]): CliArgs;
