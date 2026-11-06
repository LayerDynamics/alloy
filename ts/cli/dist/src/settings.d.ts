export type AlloySettings = {
    version: 1;
    profile: string;
    socket: string;
    db: string;
    artifact: string;
    alloyd_bin: string;
    corpus_models: string | null;
};
export declare function alloyHome(home?: string): string;
export declare function settingsPath(home?: string): string;
export declare function defaultSettings(home?: string): AlloySettings;
export declare function loadSettings(home?: string): AlloySettings | null;
export declare function saveSettings(settings: AlloySettings, home?: string): void;
