# Third-party notices

The root [MIT license](LICENSE) applies to original llm-policy-search code and
documentation, except where a file or directory carries a different license.
It does not relicense third-party material. Preserve embedded notices when
redistributing or modifying bundled material.

## Yokai environment

`src/llm_policy/environments/yokai/native/` includes an
[Apache License 2.0](src/llm_policy/environments/yokai/native/LICENSE).
That license remains in place and is included in distribution license files.
Retain any upstream copyright and attribution notices alongside the code.

## H-group Hanabi convention material

`src/llm_policy/assets/templates/hanabi/conventions/h_group_level_1/` contains
documentation and examples by the H-group convention contributors, together
with this project's offline adaptations and explanatory guides. The bundle is
licensed under [CC BY-SA 4.0](src/llm_policy/assets/templates/hanabi/conventions/h_group_level_1/LICENSE),
not MIT. The bundled README identifies the upstream revision
`a1b0e25ec3536b61dc48efe09468d534b5806962`; `source_manifest.json` records
per-file provenance, transformations, and omissions. Keep that README, manifest,
and license with the material. No upstream endorsement is implied.

## Installed dependencies

Dependencies are separately licensed and are not relicensed by this repository.
`uv.lock` records resolved versions and distribution hashes; it is not a complete
license inventory. A runtime container includes more dependencies than the
lightweight Python package. Before redistribution, review the exact wheel,
source archive, or container and retain the notices required by its contents.

## Release review

This inventory records the license files presently bundled in the repository;
it is not a claim that every historical file has verified provenance. Resolve
any additional copied-code attribution or ownership questions before release.
