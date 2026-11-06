import { type AlloySettings } from "./settings.js";
export type SetupPartial = {
    profile?: string;
    socket?: string;
    db?: string;
    artifact?: string;
    alloyd_bin?: string;
    corpus_models?: string | null;
};
export declare function runSetup(partial?: SetupPartial, home?: string): Promise<AlloySettings>;
