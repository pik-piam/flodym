# Changelog

## [2.0.0-alpha.1](https://github.com/pik-piam/flodym/compare/v1.1.0...v2.0.0-alpha.1) (2026-11-11)


### 🔖 Features

* allow missing and extra values per parameter name in parameter readers ([#221](https://github.com/pik-piam/flodym/issues/221)) ([5cd3a0a](https://github.com/pik-piam/flodym/commit/5cd3a0ad0c284658afcc7cb871d20f007e056c6e))
* attribute access for MFASystem flows, stocks, processes and parameters ([#203](https://github.com/pik-piam/flodym/issues/203)) ([92c3ed3](https://github.com/pik-piam/flodym/commit/92c3ed376dcb65747ac01b0d1b9e00300414a26e))
* auto-compute processes ([#202](https://github.com/pik-piam/flodym/issues/202)) ([418e4df](https://github.com/pik-piam/flodym/commit/418e4dfa69f229ebb26e82f55bc0429c13763a92))
* common pydantic base model forbidding unknown arguments ([#217](https://github.com/pik-piam/flodym/issues/217)) ([4dd763a](https://github.com/pik-piam/flodym/commit/4dd763aea3759286e10234a4eb48ae818cfbf5d4))
* convert stock and dims inputs, create default cohort dim ([#215](https://github.com/pik-piam/flodym/issues/215)) ([2d83ab4](https://github.com/pik-piam/flodym/commit/2d83ab410657770b9451878d552f6ba9f1fd896b))
* dynamic stock compute modes, initial stock and inflow by cohort ([#200](https://github.com/pik-piam/flodym/issues/200)) ([2c61298](https://github.com/pik-piam/flodym/commit/2c6129827b2eaecbbf7235873a7702e45c7b3f28))
* extensive flag for FlodymArray with adapted dimension handling ([#214](https://github.com/pik-piam/flodym/issues/214)) ([f002748](https://github.com/pik-piam/flodym/commit/f002748e6965c6397f151357c13688f10920b340))
* is_set property ([#124](https://github.com/pik-piam/flodym/issues/124)) ([65bed48](https://github.com/pik-piam/flodym/commit/65bed48beae92cf41f2ee5525c478c880984005a))
* new lifetime models, competing risks, period effects and mean lifetime ([#201](https://github.com/pik-piam/flodym/issues/201)) ([9fb8bec](https://github.com/pik-piam/flodym/commit/9fb8bec87c2221207d33cfa66b2fbb2e1ddfa961))
* new mass balance ([#127](https://github.com/pik-piam/flodym/issues/127)) ([b332272](https://github.com/pik-piam/flodym/commit/b332272eb60559d1b54b3ce6d2a5f9664fab65db))
* process links ([#126](https://github.com/pik-piam/flodym/issues/126)) ([4d7e68f](https://github.com/pik-piam/flodym/commit/4d7e68f8db0d804eb657b3768af46a8206b7f020))
* sub-dimensions with aggregation and disaggregation ([#213](https://github.com/pik-piam/flodym/issues/213)) ([a42f26b](https://github.com/pik-piam/flodym/commit/a42f26beb93eb7a787f46bea1d56142f496aa858))


### 🐛 Bug Fixes

* pin ty to 0.0.100 and fix its new type errors ([#222](https://github.com/pik-piam/flodym/issues/222)) ([7581543](https://github.com/pik-piam/flodym/commit/7581543deccda601c489c215e03036edf1db94fb))


### 📚 Documentation

* add migration guide from v1 to v2 ([#205](https://github.com/pik-piam/flodym/issues/205)) ([d856980](https://github.com/pik-piam/flodym/commit/d8569804c3af90d8960e706251933678d2f3d738))
* use `import flodym as fd` in examples, howtos and docstrings ([#212](https://github.com/pik-piam/flodym/issues/212)) ([0cafb90](https://github.com/pik-piam/flodym/commit/0cafb90be975ccbed3b7cd0af7863cd8dd24bbcd))


### 🧹 Miscellaneous

* configure prerelease 2.0.0-rc.1 on v2 branch ([3c9e79f](https://github.com/pik-piam/flodym/commit/3c9e79fc387c9f086500bee2ace2b2609a243c21))
* deprecated interfaces to be removed in v3.0.0 ([#206](https://github.com/pik-piam/flodym/issues/206)) ([0207ac8](https://github.com/pik-piam/flodym/commit/0207ac8ac294caff1c16e8caac2b43274dcd6a76))
* release 2.0.0-alpha.1 from main only ([#223](https://github.com/pik-piam/flodym/issues/223)) ([3c1cdc4](https://github.com/pik-piam/flodym/commit/3c1cdc40cf2cc6efd67a1a50da9a98c05600aa71))
* track test coverage in CI and add coverage badge ([#229](https://github.com/pik-piam/flodym/issues/229)) ([467e35c](https://github.com/pik-piam/flodym/commit/467e35c9a149cefdcfc0311f2b5f48bb9a74e999))
* type hints, docstrings, private names and docstring linting ([#210](https://github.com/pik-piam/flodym/issues/210)) ([63ef43b](https://github.com/pik-piam/flodym/commit/63ef43b6fac3e33cdc3f2254c3c0d9bae2f1f244))
* update pre-commit ([#181](https://github.com/pik-piam/flodym/issues/181)) ([7e9df54](https://github.com/pik-piam/flodym/commit/7e9df54427ac267ff2bf9c00cefc55b979f51adb))
* update pull request template checklist ([#216](https://github.com/pik-piam/flodym/issues/216)) ([c6dd989](https://github.com/pik-piam/flodym/commit/c6dd9895bb5933c505c5648557854d5f587dc0e2))

## [1.1.0](https://github.com/pik-piam/flodym/compare/v1.0.0...v1.1.0) (2026-10-09)


### 🔖 Features

* add typing to `data_reader` ([#186](https://github.com/pik-piam/flodym/issues/186)) ([ef4eb9e](https://github.com/pik-piam/flodym/commit/ef4eb9e81830d1cc8fa9cdb33c46bfae44043bda))
* enhance typing ([#193](https://github.com/pik-piam/flodym/issues/193)) ([d734e18](https://github.com/pik-piam/flodym/commit/d734e18191714435b53fb6c7cd3341cab630660c))
* improve type hints and accept one dimension in `get_shares_over` ([#196](https://github.com/pik-piam/flodym/issues/196)) ([1750666](https://github.com/pik-piam/flodym/commit/1750666841ce3eb12ae09f1da92c6d957d6e2710))
* lifetime extension by nature and nurture ([#192](https://github.com/pik-piam/flodym/issues/192)) ([faa44ba](https://github.com/pik-piam/flodym/commit/faa44ba26897bf928cb0da11fc054ee1105046a8))
* make dynamic stock model base public ([#190](https://github.com/pik-piam/flodym/issues/190)) ([f754db9](https://github.com/pik-piam/flodym/commit/f754db921fa236720d36872bbd18d205c0d0698b))
* small performance improvements ([#197](https://github.com/pik-piam/flodym/issues/197)) ([e81b91d](https://github.com/pik-piam/flodym/commit/e81b91d68a444e3ccf2f6dd220eff03f67434aea))


### 🐛 Bug Fixes

* keep lifetime parameters intact under lifetime extension by year, and compute it per year ([#198](https://github.com/pik-piam/flodym/issues/198)) ([5321470](https://github.com/pik-piam/flodym/commit/5321470e271c6e0092a0709c4628270ff5d5a7dd))


### 🧹 Miscellaneous

* add agents and claude config ([#188](https://github.com/pik-piam/flodym/issues/188)) ([7e7f94a](https://github.com/pik-piam/flodym/commit/7e7f94ae5eeed20b53565b00726b00ef75ad1f26))
* add type rules and checker step to CI ([#187](https://github.com/pik-piam/flodym/issues/187)) ([e5bb580](https://github.com/pik-piam/flodym/commit/e5bb5804fb72d76ad2613417d1c3605f1e2c3480))
* lifetime model - cosmetic changes ([#199](https://github.com/pik-piam/flodym/issues/199)) ([7f6eb3b](https://github.com/pik-piam/flodym/commit/7f6eb3beeb63f3489504bae83e6b24763c5f2b1b))

## [1.0.0](https://github.com/pik-piam/flodym/compare/v0.8.1...v1.0.0) (2026-08-19)


### 🔖 Features

* add a copy option to fd stocks ([413930f](https://github.com/pik-piam/flodym/commit/413930f7415d2ba8a0056c875d0b1c2bc5137ef5))
* add visualization of the abstract MFA structure ([#180](https://github.com/pik-piam/flodym/issues/180)) ([fa3d336](https://github.com/pik-piam/flodym/commit/fa3d3362e2a009185d99fc3b1efec39105f450e3))
* Complete FlodymArray type hints ([a71edd2](https://github.com/pik-piam/flodym/commit/a71edd2321a1ae0ac0a9731a5dbe42c48ee7681d))
* Convert FlodymArray to subclasses and back ([#169](https://github.com/pik-piam/flodym/issues/169)) ([d11ae27](https://github.com/pik-piam/flodym/commit/d11ae270a67342a4302b727f3ffcedd4d1dfd47d))


### 🐛 Bug Fixes

* add scalar value handling in DataFrameToFlodymDataConverter ([#162](https://github.com/pik-piam/flodym/issues/162)) ([1715113](https://github.com/pik-piam/flodym/commit/1715113e6b2a1d81367db786a28e87a1021185e4))
* enhance `Dimension.from_np` to support pandas string arrays ([#160](https://github.com/pik-piam/flodym/issues/160)) ([351c33d](https://github.com/pik-piam/flodym/commit/351c33d943f7ac3f7ac3295e5deb51334c40a824))
* mark dynamic stock model base class abstract and private ([#183](https://github.com/pik-piam/flodym/issues/183)) ([744b92d](https://github.com/pik-piam/flodym/commit/744b92d18a532cd19eb402860d0fa2f50f66a0c6))


### 📚 Documentation

* add more plots to `example1`, fix typos, add links to API ([#178](https://github.com/pik-piam/flodym/issues/178)) ([c3bb0c9](https://github.com/pik-piam/flodym/commit/c3bb0c9430b0d626bf6386872ae411683510a9f7))
* overhaul README ([d62b236](https://github.com/pik-piam/flodym/commit/d62b236b6b11c87f300e2e57fe2c62ff748bc175))


### 🧹 Miscellaneous

* add uv dependency group for development ([#161](https://github.com/pik-piam/flodym/issues/161)) ([c2f381c](https://github.com/pik-piam/flodym/commit/c2f381ca478926a6a779aedd018e92e09695439b))
* fix (random) test failures due to tkinter ([#177](https://github.com/pik-piam/flodym/issues/177)) ([a72d433](https://github.com/pik-piam/flodym/commit/a72d433a8c0fcf4e73e948471bd0c1062b483a64))
* init release-please version to 0.8.1 ([#171](https://github.com/pik-piam/flodym/issues/171)) ([6f1f5b2](https://github.com/pik-piam/flodym/commit/6f1f5b2d1a6b9b9b0d769024c7e24e5661f4e87a))
* migrate to ruff  ([#167](https://github.com/pik-piam/flodym/issues/167)) ([96188af](https://github.com/pik-piam/flodym/commit/96188af8e0eb8fca33535246471829bfae5191b6))
* only run release PR job in main repo ([#179](https://github.com/pik-piam/flodym/issues/179)) ([792190d](https://github.com/pik-piam/flodym/commit/792190d6eb9adc2787ef452ce2ba6362de4359ce))
* pre-commit autoupdate ([#157](https://github.com/pik-piam/flodym/issues/157)) ([b7365ca](https://github.com/pik-piam/flodym/commit/b7365ca46d02a255f24ef0b1603982c40014ef06))
* remove unnecessary main guard from test files ([b181610](https://github.com/pik-piam/flodym/commit/b18161078cf3aa64e9fce1a22495f06f438531ed))
* target 1.0.0 ([c665d6f](https://github.com/pik-piam/flodym/commit/c665d6f9d08847f29cac8cfb4407b8961c391800))
* target 1.0.0 ([#176](https://github.com/pik-piam/flodym/issues/176)) ([08fcf64](https://github.com/pik-piam/flodym/commit/08fcf640d0bb7259b20b6c289bed25e0cd1fda43))
* update `uv.lock` file after version increment ([#175](https://github.com/pik-piam/flodym/issues/175)) ([832e39d](https://github.com/pik-piam/flodym/commit/832e39d2a33cce030c9e40dad00547f96ee3d4f7))
* update build system to use uv_build ([#163](https://github.com/pik-piam/flodym/issues/163)) ([9032466](https://github.com/pik-piam/flodym/commit/90324662c8ca98ed34ed6991e65cf6a75b1faab9))
* update GitHub Actions workflow to use uv ([#165](https://github.com/pik-piam/flodym/issues/165)) ([ef55ab6](https://github.com/pik-piam/flodym/commit/ef55ab69bd7cfad940f9b094f5518a3656158876))
* update pre-commit ([#173](https://github.com/pik-piam/flodym/issues/173)) ([c9ff4b5](https://github.com/pik-piam/flodym/commit/c9ff4b5c1e5af130bd7afefa595b385d491b6198))
* use conventional commits for pre-commit ([#168](https://github.com/pik-piam/flodym/issues/168)) ([807eee8](https://github.com/pik-piam/flodym/commit/807eee8993d6c8d8b537e0f6b11d5a96c4b84067))
* use release-please to generate release PRs ([#166](https://github.com/pik-piam/flodym/issues/166)) ([d3fec89](https://github.com/pik-piam/flodym/commit/d3fec8910cd93d63d9ec4e595fdd97266e86fcd4))
