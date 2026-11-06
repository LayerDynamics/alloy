export type SessionEvent = {
    session_id: string;
    event: string;
    body: Record<string, unknown>;
};
export declare class RpcClient {
    #private;
    private constructor();
    static connect(socketPath: string): Promise<RpcClient>;
    call(method: string, params?: Record<string, unknown>): Promise<unknown>;
    callStreaming(method: string, params: Record<string, unknown>, onEvent: (event: SessionEvent) => void): Promise<unknown>;
    close(): void;
}
