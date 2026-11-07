import { existsSync, readFileSync } from "node:fs";

/** Read the installed package's version, independently of the current directory. */
export function cliVersion(): string {
  const sourceMetadata = new URL("../package.json", import.meta.url);
  const metadataPath = existsSync(sourceMetadata)
    ? sourceMetadata : new URL("../../package.json", import.meta.url);
  const metadata: unknown = JSON.parse(readFileSync(metadataPath, "utf8"));
  if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)
      || !("name" in metadata) || metadata.name !== "alloy"
      || !("version" in metadata) || typeof metadata.version !== "string"
      || !/^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/.test(metadata.version)) {
    throw new Error("Alloy package metadata has no valid version. Reinstall Alloy using your installation method.");
  }
  return metadata.version;
}
