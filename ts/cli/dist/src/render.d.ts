export type JsonRpcError = {
    code: number;
    message: string;
    data?: {
        kind?: string;
        [key: string]: unknown;
    };
};
export declare function textFromEvent(line: {
    method?: string;
    params?: {
        event?: string;
        body?: {
            text?: string;
        };
    };
}): string | null;
export declare function formatRpcError(error: JsonRpcError): string;
