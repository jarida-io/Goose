# PCTX protocol and telemetry compatibility patches

These sources retain the optional Goose code-mode integration while removing
its older RMCP and OpenTelemetry dependency instances. Disabling code-mode
would not fix its resolved optional dependencies.

| Package | Registry version | Upstream source commit |
| --- | --- | --- |
| pctx_code_mode | 0.4.2 | cee9b86add007cf68f0eaf06dece1f0ca00ece63 |
| pctx_config | 0.1.5 | a3e69e40e3403ed6544e5495287e824261e7d5b0 |
| pctx_registry | 0.1.2 | 998fa8c1b48fbcbba21b103dbb9db07d8b2a2187 |
| pctx_code_execution_runtime | 0.2.0 | ba88ffa77be7331ff9bd18dea5fc4fadcd8578c7 |

Source: https://github.com/portofcontext/pctx. Each directory retains the MIT
license. Cargo manifests are the published normalized manifests. Registry
checksums and generated local lockfiles are omitted because these are explicit
path patches, not replacements in the registry cache.

Changes are limited to RMCP 2.1 dependency requirements, the OpenTelemetry
0.32 release family with SDK >=0.32.1, and flattening registry content handling
to the RMCP ContentBlock type. Code-mode also accepts the resolved code
generator's infallible Tool constructor; schema parsing still returns errors. No code-mode feature, permission, or sandbox
boundary is removed. The execution runtime's generated JavaScript is retained
unchanged. Consumers of Goose as a path dependency must mirror these four
patch declarations in their workspace root.

Retire the patches when upstream PCTX releases compatible versions. Verify
both the normal application build and the optional code-mode feature before
removing or updating them.
