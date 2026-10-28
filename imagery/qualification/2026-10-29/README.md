# Recent imagery qualification — 29 October 2026

**Subsequent decision:** the user chose the approved 2015 CORE3D pilot for the historical end-to-end demonstration. Recent source selection is deferred. This report preserves the earlier search findings; its unresolved gate does not block the active historical pilot. See [the current plan](../../../docs/PLAN_V2.md).

**Result: no dataset qualified for reconstruction.** The user selected free sources, permitted a different pilot location, and confirmed no research affiliation or existing research-programme access. Recent raw imagery with usable RPC cameras, color, sufficient overlapping acquisitions, and independent validation remains unresolved. This review does not establish that such a free dataset is unavailable globally.

[Machine-readable result](result.json), [candidate AOI](pilot-aoi.geojson), and [evidence checksums](evidence/sha256.json) preserve the findings. Evidence records include the exact query URL and retrieval time. No acquisition has been selected for training or holdout.

## Requirements used

- Initial candidate: the existing Jacksonville block centered at longitude −81.66376054127814°, latitude 30.34881972282938°, 250 m square plus a 20 m halo. The GeoJSON uses geodesic diagonals to approximate metric squares. Another location is allowed, but has not been selected.
- Provisional capture window: 2025-10-29 through 2026-10-29. This advances the preceding-12-month planning window by one day; it is not a newly approved hard age threshold.
- At least three training acquisitions and one independent acquisition holdout, preferably within seven days. Multiple tiles from one acquisition do not count as extra views. The historical 13-day exception does not apply.
- High-resolution sensor imagery with usable RPCs and matched color, suitable viewing geometry, and usage rights compatible with reconstruction and intended derived outputs.
- Date-compatible independent height/control evidence, or documented unchanged validation areas with the resulting limitation.

## Jacksonville catalogue check

The public Airbus [Pléiades Neo all-segments layer](https://portal.intelligence-airbusds.com/arcgis/rest/services/OneAtlas/OADConstellationArchive/MapServer/4) and [Pléiades all-segments layer](https://portal.intelligence-airbusds.com/arcgis/rest/services/OneAtlas/OADConstellationArchive/MapServer/9) returned these records for a bounding-box intersection query in the capture window:

| Acquisition identifier | Capture time (UTC) | Reported incidence | Reported cloud | Covers halo bounding box? |
| --- | --- | --- | --- | --- |
| `DS_PHR1A_202511111613124_FR1_PX_W082N30_0407_06110` | 2025-11-11 16:13:12 | 19.06° | 0% | Yes |
| `ACQ_PNEO3_06504903263900` | 2025-11-20 16:08:58 | 29.72° | 0% | No; misses the block center |
| `ACQ_PNEO3_07012305023853` | 2026-04-04 16:05:54 | 20.90° | 0% | Yes |

Coverage was checked against the returned single convex quadrilateral footprints and all four halo bounding-box corners. Neither query reported `exceededTransferLimit`. These are catalogue footprints and scene-level cloud estimates; usable pixels and local cloud conditions have not been inspected. The public preview's completeness and freshness are unknown. No RPC delivery, price, free access, derivative licence, or independent validation has been verified. The records do not form the required group even if combined across sensors.

The [Vantor Discovery](https://developers.maxar.com/docs/discovery/guides/discovery-guide) request returned HTTP 401 without credentials. Its archive coverage is unknown, rather than absent. Commercial archive/tasking options are outside the selected free-only search. Earlier pricing research is not a purchase recommendation.

## Free-source findings

| Source | Evidence | Decision for this pilot |
| --- | --- | --- |
| Maxar Open Data, recent disaster releases | Public [event catalogue](https://maxar-opendata.s3.amazonaws.com/events/catalog.json) retrieved. The [Sri Lanka collection](https://maxar-opendata.s3.amazonaws.com/events/Cyclone-Ditwah-Sri-Lanka-Nov-2025/collection.json) includes recent acquisitions. A 2025-12-05 WV01 item was inspected below. | Sample fails the raw camera/color contract. Other acquisitions have not been exhaustively checked; no replacement AOI or qualifying group selected. |
| Satellogic EarthView | The provider's [February 2025 announcement](https://satellogic.com/news/press-releases/satellogic-earthview-dataset-now-openly-accessible-via-registry-of-open-data-on-aws/) explicitly dates its imagery to July–December 2022. | Release date does not establish recent capture. A newer release with qualifying dates and RPC delivery has not been verified. |
| Pléiades Glacier Observatory | [PGO](https://www.legos.omp.eu/pgo/) advertises recent glacier campaigns and conditional access for science users; its [product description](https://www.legos.omp.eu/pgo/pgo-products/) describes stereo imagery with RPCs. The product page could not be retrieved during live verification. | Research lead only. Eligibility, licence, actual recent files, four-view group, color, validation, and suitability for an urban pilot are unverified. No affiliation does not by itself prove ineligibility. |
| DINAMIS / Pléiades World Heritage | [PWH](https://dinamis.data-terra.org/en/pleiades-world-heritage/) offers sensor processing options to authorized users, subject to eligibility and its specific licence. | No existing access; not an unrestricted free download. Eligibility and any cost must be established before treating this as an input source. |
| NASA CSDA | [NASA's FAQ](https://science.nasa.gov/earth-science/csda/faqs/) requires authorization under the applicable vendor/product licence tier. | No existing access or affiliation; eligibility is unverified. Not an immediately available dataset. |
| ISRO Bhoonidhi | [NRSC ordering guidance](https://www.nrsc.gov.in/nrscnew/archive_order_procedure.php?lang_code=en) distinguishes open data at 5 m GSD and coarser from finer-resolution ordering. | The general free tier does not meet the proposed sub-meter urban input target. Free sample availability is not proof of a recent four-view dataset. |

### Bounded Maxar image-header inspection

Inspected [item `102001011D2F3500`, tile `033313300132`](https://maxar-opendata.s3.amazonaws.com/events/Cyclone-Ditwah-Sri-Lanka-Nov-2025/ard/44/033313300132/2025-12-05/102001011D2F3500.json), captured 2025-12-05 08:00:38 UTC. Its listed assets are a data mask, panchromatic image, and visual image; no RPC sidecar is linked.

HTTP range requests retrieved only the first 262,144 bytes of each TIFF (524,288 bytes total). Rasterio opened both header excerpts without reading pixels. Both report EPSG:32644, one band, zero RPC tags, and `rpcs is None`. See [range responses](evidence/header-downloads.json) and [inspection results](evidence/header-inspection.json). The excerpts remain under ignored `artifacts/imagery/qualification/2026-10-29/`.

This item fails the existing raw RGB+RPC input contract. Its resampled visual pixel spacing must not be treated as native multispectral resolution. These findings apply to this sample, not every asset in Maxar's catalogue. The [AWS registry](https://registry.opendata.aws/maxar-open-data/) describes the programme's ARD products and CC BY-NC 4.0 terms; free access does not establish suitability for commercial derived releases.

## Independent validation

The exact-AOI USGS [TNMAccess](https://tnmaccess.nationalmap.gov/api/v1/docs) LPC query returned two tiles: `FL_DUVALCO_2007` and `FL_Peninsular_FDEM_2018_D19_DRRA`. See [response](evidence/usgs-lpc.json). Those are older project identifiers; exact acquisition intervals and vertical datums still require source metadata. Publication and last-update dates do not establish acquisition dates. Neither is accepted as contemporary ground truth. This does not rule out newer data in another catalogue or location.

## Next action and stop condition

The search actions below are deferred by the subsequent 2015 decision above.

Continue only with a public sensor-data candidate or an access route whose eligibility can be established without assuming an affiliation. Use the [request brief](request-brief.md) to check a candidate before downloading bulk imagery. A different AOI alone does not resolve missing camera models.

If no recent free group can be verified, explicitly decide whether to relax capture age for a historical demonstration. Until then, preserve the 2015 fixture for regression, keep the current-city gate open, and do not launch a long training run or label historical output as current. No purchases, account registrations, access applications, or external messages were made.
