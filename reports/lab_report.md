# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** HVNhan-Relieq
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19–20/08/2026

> **Môi trường chạy thật (khác đề bài, đã kiểm chứng bằng API):** Neo4j 5.26 Community chạy native `bolt://localhost:7687` (instance AuraDB trong `.env` đã bị xoá, DNS không phân giải); extraction + generator dùng Groq `openai/gpt-oss-*` (model `llama-3.3-70b-versatile` của đề bài **đã bị Groq gỡ**); judge `google/gemma-4-31b-it` trên NVIDIA NIM (key OpenRouter trong `.env` hết credit). Chi tiết từng sai lệch nằm ở bảng đầu notebook.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 0. Đặc thù dữ liệu quyết định toàn bộ thiết kế

Trước khi trả lời các câu hỏi, phải nêu một sự thật về dump này vì nó chi phối mọi kết quả phía sau: **`HackerNoon/tech-company-news-data-dump` không chứa nội dung bài báo**, chỉ có `title` + `description` (~32 từ). Trong 5.000 dòng đầu:

| Chỉ số | Giá trị |
|---|---|
| Dòng stream về | 5.000 |
| Có `description` và đủ dài (≥ 80 ký tự) | 2.686 |
| Sau exact dedup (SHA-1 trên text đã normalize) | **2.113** |
| Số chunk sinh ra (220 từ/chunk) | 2.113 (≈ 1 chunk/bài) |
| Độ dài trung bình | 42,3 từ/chunk |

Hệ quả trực tiếp: **một chunk gần như không bao giờ chứa đủ 2 hop**. Mọi quan hệ multi-hop buộc phải nối qua các thực thể trùng nhau **giữa các bài khác nhau** — đây chính là thứ GraphRAG được kỳ vọng làm tốt hơn Flat RAG, nhưng cũng có nghĩa chất lượng đồ thị phụ thuộc hoàn toàn vào độ phủ của bước extraction.

---

### 1. Coreference Resolution (Phân giải đại từ)

*Trả lời:*

- **Ví dụ từ dữ liệu (`r00077::c0000`):**
  - *BEFORE:* "Revolut to stop crypto services for U.S. customers. LONDON Aug 4 (Reuters) - British fintech Revolut will stop allowing U.S. customers to access cryptocurrencies **the company** said in a statement on Friday…"
  - *AFTER:* "…access cryptocurrencies **Revolut** said in a statement on Friday…"
  Đây là ca resolve **đúng** vì chỉ có một pháp nhân trong chunk.

- **Hiện tượng nguy hiểm nhất trong dump này** là dạng snippet có **hai công ty trong một câu**: *"X announces partnership with Y. **The company** will provide…"*. `the company` có thể là X hoặc Y, và không có ngữ cảnh nào trong chunk để phân định — đúng loại tình huống prompt được yêu cầu **không đoán**.

- **Kết quả định lượng của prompt bảo thủ:** chỉ **88/260 chunk (33,8%)** bị chỉnh sửa, và **18 mention được log vào `unresolved_mentions`** thay vì bị resolve bừa. Nhóm không resolve được nhiều nhất là `we`, `they`, `you`, `it` — hợp lý, vì phần lớn dump là **thông cáo báo chí viết ở ngôi thứ nhất**, `we` không hề có antecedent trong chunk nên mọi phép resolve đều là bịa.

- **Hậu quả nếu resolve sai:** tạo **false edge có provenance hợp lệ**. Đây là loại lỗi tệ nhất trong pipeline: cạnh sai vẫn mang `source_chunk_id` và `published_date` đầy đủ nên mọi sanity check đều pass, và chỉ có thể phát hiện bằng cách đọc lại chunk gốc. Ví dụ: nếu `the company` trong câu partnership bị gán nhầm, ta tạo cạnh `PARTNERED_WITH` hoặc `DEVELOPED` cho **sai pháp nhân** — nghĩa là đồ thị nói dối một cách có bằng chứng.

---

### 2. Entity Resolution Threshold & Lexical Guard

*Trả lời:*

- **Ngưỡng cosine similarity:** `ER_THRESHOLD = 0.90`, **không phải 0.85**. Lý do: trong không gian `all-MiniLM-L6-v2`, tên tổ chức ngắn có similarity nền rất cao — hai công ty công nghệ bất kỳ thường đã đạt 0.6–0.8 chỉ vì cùng "hình dạng" tên. Ngưỡng thấp biến ER thành cỗ máy false merge.

- **Vector similarity không đủ, phải có guard.** Điểm mấu chốt trong thiết kế của tôi: **vector chỉ sinh candidate, không được ra quyết định**. Quyết định do 5 luật lexical đưa ra, mỗi luật kèm lý do ghi vào audit:

| Luật | Cặp bị chặn | Vì sao vector sai |
|---|---|---|
| `PERSON_FIRSTNAME` | `Sam Altman` vs `Steve Altman` | `SequenceMatcher` = **0.78**, vượt ngưỡng lexical 0.72 → nếu chỉ dựa ratio là gộp nhầm 2 người |
| `DIGIT_MISMATCH` | `GPT-4` vs `GPT-3` | Embedding coi chữ số như nhiễu, similarity ~0.97, nhưng là 2 phiên bản sản phẩm khác nhau |
| `TOKEN_SUPERSET` | `Apple` vs `Apple Watch` | Tên sản phẩm chứa trọn tên công ty |
| `SUFFIX_ONLY` (cho gộp) | `Microsoft Corp` ≡ `Microsoft` | Chỉ khác hậu tố pháp lý |
| `CORP_TOKEN_EXTRA` (cho gộp) | `Meta Platforms` ≡ `Meta` | Token dư là token doanh nghiệp đã biết |

  Cả 5 ca này được đóng thành **unit test chạy trong notebook** (`_guard_test_df`, assert 5/5 pass), nên guard không thể âm thầm hỏng khi ai đó chỉnh code.

---

### 3. Đồ thị & Super-node Mitigation

*Trả lời:*

- **Ngưỡng và chính sách:** `SUPER_NODE_DEGREE = 100` → cắt còn `SUPER_NODE_EDGE_CAP = 50` cạnh mới nhất; `GLOBAL_EDGE_CAP = 250` cho toàn truy vấn. `LIMIT` được đặt **bên trong Cypher**, không phải cắt ở client — nếu cắt ở client thì super-node vẫn kéo toàn bộ cạnh qua mạng, đúng thứ cần tránh.

- **Ưu điểm của cắt tỉa theo `published_date` mới nhất:** tin công nghệ mang bản chất *event-state* — "to acquire" hôm nay bị thay bởi "has acquired" tháng sau. Khi buộc phải cắt, giữ bản mới nhất là mặc định đúng cho câu hỏi về trạng thái hiện tại.

- **Rủi ro đối xứng (và nó không hề lý thuyết):** chính Golden Dataset có câu `G5000-02` yêu cầu so sánh *planned* (row 33, 1746) với *completed* (row 935). Với câu này, cắt theo "mới nhất" sẽ **xoá đúng bằng chứng cũ cần để đối chiếu**. Vì vậy context vẫn giữ nguyên trường `date` để LLM tự nhận ra khoảng trống thời gian, và `SUPER_NODE_EDGE_CAP` được đặt đủ rộng (50). Hướng đúng hơn cho production là **cắt tỉa theo truy vấn** (ưu tiên relation type liên quan tới câu hỏi) chứ không chỉ theo thời gian.

- **Thực tế đo được trên đồ thị lab này:** `max_degree = 8`, `p95_degree = 3`, **0 node vượt ngưỡng 100**. Vì vậy nói "đã xử lý super-node" mà không có ca thật là vô nghĩa. Cell 5.1 xử lý theo 2 nhánh và in rõ đang ở nhánh nào: không có ca thật thì **hạ ngưỡng xuống p95 thật** để kích hoạt đúng nhánh cắt tỉa, kiểm tra `len(edges) <= 50` **và** assert các cạnh lấy về đúng là các cạnh mới nhất (so sánh danh sách `published_date` đã sắp xếp). Tức là kiểm chứng *cơ chế*, và nói thẳng rằng đó là test cơ chế.

- **Top thực thể theo degree (đồ thị thật, không phải ví dụ):**

| Hạng | Tên thực thể | Loại | Degree |
|---|---|---|---|
| 1 | ServiceNow | Company | 8 |
| 2 | Microsoft | Company | 5 |
| 3 | OpenAI | Company | 4 |

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

**Thiết lập:** 12 câu lấy mẫu phân tầng từ Golden 50 câu (4 factoid · 4 multi-hop · 4 cross-doc), cùng embedder, cùng generator (`gpt-oss-20b`), cùng prompt trả lời; judge là model độc lập `google/gemma-4-31b-it` chạy trên NVIDIA NIM.

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge)

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ | Nhận xét phân tích |
|---|---|---|---|---|
| **Comprehensiveness (1–5)** | 4.333 | **5.000** | +0.667 | Toàn bộ chênh lệch đến từ 2 câu Flat RAG bị 1 điểm; xem phân tích ca lỗi bên dưới |
| **Faithfulness (1–5)** | 5.000 | 5.000 | 0.000 | Cả hai đều không bịa — Flat RAG khi thiếu dữ kiện thì **nói thẳng là không có**, nên vẫn faithful |
| **Multi-hop Reasoning (1–5)** | 4.500 | **5.000** | +0.500 | |
| **Latency trung bình (s)** | 0.794 | 1.221 | +0.427 | GraphRAG chậm hơn **1.54×**: thêm 1 lượt LLM trích seed + các round-trip Cypher của BFS |
| **Token usage trung bình** | 797.5 | 1373.4 | +575.9 | GraphRAG tốn **1.72×**: context = subgraph + vector chunks (superset của Flat RAG) |

**Theo nhóm câu hỏi** (nguồn: `outputs/graphrag_vs_flatrag_summary.csv`):

| Nhóm | Comprehensiveness (Flat → Graph) | Multi-hop (Flat → Graph) | Token |
|---|---|---|---|
| factoid | 5.000 → 5.000 (**hòa**) | 5.000 → 5.000 | 1.72× |
| multi-hop | 4.000 → **5.000** | 4.500 → **5.000** | 2.05× |
| cross-doc | 4.000 → **5.000** | 4.000 → **5.000** | 1.42× |

Kết quả khớp đúng kỳ vọng lý thuyết: **factoid hòa** (một chunk là đủ, đồ thị không thêm gì), **multi-hop và cross-doc là nơi GraphRAG kiếm điểm**, và cái giá phải trả là token/latency.

> ⚠️ **Hạn chế phải nói rõ:** judge `gemma-4-31b` **rất dễ dãi** — 24/24 lượt chấm `faithfulness` đều là 5, và GraphRAG đạt 5 tuyệt đối ở mọi câu. Có hiệu ứng trần (ceiling effect), nên các con số trên chỉ nói được "GraphRAG không thua ở đâu và thắng ở 2 câu Flat RAG trượt", **không** đủ để kết luận mức độ vượt trội. Judge `deepseek-v4-flash` chấm khắt khe hơn hẳn (cùng một câu: 3/5/1 thay vì 5/5/5) nhưng có độ trễ ~240 giây/lượt trên NIM nên không khả thi cho 24 lượt chấm trong khung giờ lab. Với mẫu 12 câu, đây là **chỉ báo, không phải kết luận có ý nghĩa thống kê**.

#### Phân tích 2 Ca lỗi Điển hình

**1. Flat RAG thất bại — GraphRAG thành công: `G5000-34` (multi-hop)**

- *Câu hỏi:* "Compare how Google Cloud and Amazon expanded their AI ecosystems… Which third-party model/technology suppliers are named for each?"
- *Điểm:* Flat RAG **1**/5/3 · GraphRAG **5/5/5**
- *Tại sao Flat RAG thất bại:* câu hỏi cần **hai nhánh dữ kiện** (Google Cloud và Amazon) nằm ở hai bài khác nhau. Top-k=6 theo cosine bị **một phía chiếm trọn**: mọi chunk trả về đều nói về Amazon. Judge ghi đúng bản chất: *"failed to provide any information about Google Cloud… However, it is perfectly faithful to the provided context, which indeed contains no information about Google Cloud."* Đây là **failure mode kinh điển của vector-only**: similarity là quan hệ *một chiều giữa câu hỏi và từng chunk*, không hề có cơ chế bảo đảm **độ phủ các nhánh** của câu hỏi.
- *GraphRAG giải quyết thế nào:* seed extraction tách câu hỏi thành 2 thực thể (`Google Cloud`, `Amazon`), match được cả hai trong Neo4j, rồi BFS thu 8 cạnh quanh **cả hai** seed. Cấu trúc đồ thị ép context phải chứa cả hai nhánh — thứ mà top-k không bao giờ đảm bảo. Ca `G5000-32` (OpenAI plug-in tháng 3 vs app-store tháng 6) hỏng theo đúng cơ chế đó và cũng được GraphRAG cứu.

**2. GraphRAG "thất bại" ở tầng retrieval nhưng vẫn đúng nhờ hybrid: `G5000-04` (cross-doc)**

- *Điểm:* cả hai **5/5/5** — nhưng chẩn đoán cho thấy `graph_no_seed = 1`, `seeds_matched = 0`, `edges_collected = 0`: **subgraph rỗng hoàn toàn**.
- *Nguyên nhân gốc:* câu hỏi nói về *"two named Ericsson IoT businesses"* — tức **IoT Accelerator** và **Connected Vehicle Cloud**. Đây là **đơn vị kinh doanh**, không thuộc bất kỳ loại nào trong schema allowlist (`Company | Person | Technology`), nên bước NER/RE **không bao giờ tạo node cho chúng**. Seed extraction sinh ra tên đúng nhưng Neo4j không có gì để match → NO_SEED.
- *Vì sao vẫn đạt 5 điểm:* nhánh **vector trong hybrid** đã cứu. Đây chính là lý do `answer_graph_rag()` giữ lại top-k vector thay vì graph-only: khi seed matching trượt, hệ thống **suy giảm về Flat RAG** thay vì trả lời rỗng.
- *Đề xuất khắc phục:* (a) mở rộng schema thêm `Product`/`BusinessUnit` — đúng nhưng làm tăng nhiễu ER; (b) khi NO_SEED thì **tự động nâng k của nhánh vector** để bù ngân sách context đang bị bỏ phí; (c) ghi `graph_no_seed` thành metric giám sát — một hệ GraphRAG production mà tỉ lệ NO_SEED cao thì đang trả tiền cho đồ thị mà không dùng.

---

### 4b. Bổ sung: hai phát hiện về chính công cụ đo

**Judge sao chép giá trị mẫu trong prompt.** Bản prompt chấm điểm đầu tiên của tôi kết thúc bằng template `{"comprehensiveness": 1, "faithfulness": 1, "multi_hop_reasoning": 1, "rationale": "..."}`. Judge **copy nguyên số 1** trong khi `rationale` lại khen câu trả lời — tạo ra một bảng benchmark chỉ gồm 1 và 5, nhìn vẫn "hợp lý" (GraphRAG thắng multi-hop +2.0) nhưng hoàn toàn là rác. Phát hiện được nhờ đối chiếu điểm với rationale. **Bài học: trong prompt chấm điểm, chỉ mô tả KIỂU dữ liệu, không bao giờ đưa GIÁ TRỊ mẫu.** Parser cũng đã được viết lại để `raise` khi không tìm thấy điểm, thay vì lặng lẽ trả về mặc định 1 — lỗi đo lường không được phép biến thành dữ liệu.

**Ngưỡng ER 0.90 gây mất recall thật.** Audit ghi cặp `Synopsys` vs `Synopsys Inc.` ở similarity **0.8324 → REJECT_THRESHOLD**. Đây là cặp **đáng lẽ phải gộp** (chỉ khác hậu tố pháp lý, guard `SUFFIX_ONLY` sẽ cho gộp nếu được xét). Nguyên nhân: `all-MiniLM-L6-v2` hạ similarity đáng kể khi thêm token `Inc.`. Điều này chỉ nhìn thấy được vì tôi **tách ngưỡng ghi audit (0.80) khỏi ngưỡng quyết định gộp (0.90)** — nếu audit chỉ ghi các cặp đã gộp thì không bao giờ kiểm toán được thứ mình bỏ sót. Hướng sửa đúng: **chuẩn hoá bỏ hậu tố TRƯỚC khi embed**, để cặp này trở thành trùng khít ở tầng lexical thay vì trông cậy vào vector.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent

*Trả lời:*

- **Đánh đổi Quality vs Cost vs Latency:**
  GraphRAG trả thêm chi phí ở **ba chỗ**, không chỉ một:
  1. *Indexing overhead (trả trước, một lần):* coreference cho 260 chunk tốn **~25 phút** wall-clock dưới trần 8.000 token/phút của Groq free tier (NER/RE thêm ~2,5 phút nữa khi đã chạy song song 2 luồng). Flat RAG chỉ cần embed 2.113 chunk bằng MiniLM local — vài chục giây, 0 đồng.
  2. *Latency mỗi truy vấn:* GraphRAG cần **1 lượt gọi LLM phụ để trích seed entity** trước khi chạm database, cộng với N truy vấn Cypher cho BFS (mỗi node expand là 2 round-trip: `node_degree` rồi `recent_edges`).
  3. *Token mỗi truy vấn:* context của GraphRAG = subgraph + vector chunks, tức là **superset** của context Flat RAG.
  Đổi lại, chỉ GraphRAG mới có thứ Flat RAG không thể có: quan hệ đã được chuẩn hoá kèm `published_date` và `source_chunk_id`, cho phép trả lời "trạng thái sự kiện thay đổi thế nào theo thời gian" mà không phụ thuộc vào việc top-k vector có tình cờ gắp đủ các bản tin hay không.

- **Quyết định từ chối AI Coding Agent:**
  1. **Từ chối xoá near-duplicate.** Agent đề xuất pipeline dedup kinh điển: phát hiện near-dup rồi *drop* bản trùng để tiết kiệm token. Tôi từ chối vì Golden Dataset của lab **dựa vào chính các bản gần trùng**: câu `G5000-02` yêu cầu so sánh *"to acquire"* (row 33, 1746) với *"has acquired"* (row 935). Bằng chứng định lượng: row 33 và row 1746 có **Jaccard trên tiêu đề = 1.000** — dedup theo tiêu đề, lối tắt tự nhiên nhất, sẽ xoá mất một nửa cặp đối chiếu; trên full text chúng chỉ đạt 0.419. Kết luận: **FLAG (gắn `near_dup_group_id`), không DROP**, và chỉ dùng group để giới hạn dư thừa *trong context*, không đụng vào corpus.
  2. **Từ chối pairwise cosine O(N²)** cho near-dedup và cho entity resolution. Với 2.113 bài, pairwise là 2.231.328 cặp; MinHash+LSH banding chỉ sinh **304 candidate pair** (giảm 99,99%) rồi mới verify Jaccard thật trên đúng 304 cặp đó.
  3. **Từ chối hạ ngưỡng entity resolution xuống 0.85.** Trong không gian `all-MiniLM-L6-v2`, tên tổ chức ngắn có similarity nền rất cao nên 0.85 gộp nhầm hàng loạt; tôi giữ 0.90 **và** thêm 5 luật lexical guard, vì ngưỡng vector một mình không phân biệt được `Sam Altman`/`Steve Altman` (ratio 0.78) hay `Apple`/`Apple Watch`.
  4. **Từ chối để LLM tự do đặt tên quan hệ.** `type(r)` không tham số hoá được trong Cypher nên chuỗi relation bị **nội suy thẳng vào query**; nếu tin LLM, một relation dạng `` FOO]->(x) WITH x MATCH (n) DETACH DELETE n // `` sẽ chạy thật. Allowlist ở đây là **kiểm soát an ninh**, không chỉ là kỷ luật schema.
  5. **Từ chối chỉ trích xuất đúng 51 chunk chứa evidence.** Làm vậy đồ thị sẽ "đẹp" trên Golden Dataset nhưng là teaching-to-the-test: mọi seed đều trúng vì graph chỉ chứa đúng thứ được hỏi. Ngân sách 260 chunk = 51 evidence + 209 mẫu ngẫu nhiên seed 42, và tỉ lệ phủ ~12% được ghi thẳng vào phần threat-to-validity.

- **Giải pháp scale 350MB (~100.000 bài):**
  Bottleneck đầu tiên **không phải Neo4j mà là throughput LLM cho bước extraction** — đo được ngay trong lab này: trần 8.000 token/phút **và trần 200.000 token/ngày (tính RIÊNG cho từng model)** biến 260 chunk thành ~28 phút. Ngoại suy tuyến tính: 100.000 chunk ≈ **180 giờ** nếu tính theo trần phút, và tệ hơn nhiều nếu tính theo trần ngày — chỉ riêng extraction đã cần ~3,8 triệu token, tức **19 ngày** trên hạn mức 200k/ngày của một model. Thứ tự xử lý:
  1. *Tăng song song có kiểm soát:* worker pool async + token-bucket dùng chung (cơ chế `TokenBucket` trong notebook đã là bản một tiến trình của việc này), chuyển sang tier trả phí hoặc self-host để nâng trần TPM.
  2. *Giảm khối lượng phải trích xuất:* lọc trước bằng NER rẻ (spaCy/regex) để bỏ chunk không chứa thực thể tổ chức nào — với dump này ~46% bản ghi rỗng nội dung và ~27% còn lại không sinh triple nào, tức phần lớn ngân sách LLM đang đốt vào chunk vô ích.
  3. *Bottleneck thứ hai — entity resolution:* FAISS `IndexFlatIP` là O(N) mỗi truy vấn; ở quy mô hàng triệu mention phải chuyển sang **HNSW + blocking theo ký tự đầu/token đầu** để không phải so mọi mention với mọi mention.
  4. *Bottleneck thứ ba — super-node:* ở 100.000 bài, các thực thể như `Microsoft` chắc chắn vượt degree 100 (trong lab này chưa vượt vì đồ thị quá nhỏ). Lúc đó cắt tỉa theo `published_date` là chưa đủ — cần index trên `(degree, published_date)` và cân nhắc tách quan hệ theo cửa sổ thời gian, kèm community partitioning để truy vấn vĩ mô không phải duyệt toàn đồ thị.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module | Hàm / Khối code | Quan sát thực tế & Đánh giá |
|---|---|---|---|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()`, `COREF_SYSTEM` | Chỉ 33,8% chunk (88/260) bị chỉnh sửa — đúng tinh thần bảo thủ. 18 mention được log là *không* resolve được (chủ yếu `we`, `they`, `you`, `it`), tức prompt đã chịu "bó tay có kiểm soát" thay vì đoán bừa. Đáng chú ý: `we`/`you` xuất hiện nhiều vì dump chứa thông cáo báo chí viết ở ngôi thứ nhất — loại này **không có antecedent trong chunk**, resolve là bịa. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS`, vòng lọc trong `run_extraction()` | Allowlist chặn ở tầng client trước khi chuỗi relation được nội suy vào Cypher. Đây là ranh giới an ninh thật, không phải hình thức: `type(r)` không parameterise được. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()`, `batches()` | `UNWIND $rows AS row` theo batch 1000, nhóm theo label/relation type. `MERGE` trên `{source_chunk_id}` khiến việc nạp là idempotent ở cấp cạnh; thêm `reset_graph()` để chạy lại notebook không nhân đôi số liệu. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `merge_guard()`, `UF` | Union-Find biến quan hệ "gộp từng cặp" thành cụm nhất quán (nếu A≡B và B≡C thì A,B,C cùng canonical). Guard chạy **trước** `union()` nên một cặp bị chặn không thể lọt vào cụm qua đường vòng. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `recent_edges()` | `LIMIT` nằm **trong Cypher**, không phải cắt ở client — nếu cắt ở client thì super-node vẫn kéo hết cạnh về qua mạng, đúng thứ cần tránh. |
| **LLM-as-a-Judge** | Module 5 | `judge_answer()`, `judge_json()` | Judge chạy model khác hẳn generator. Điểm `faithfulness` được chấm **trên context của chính hệ thống đó**, nên một câu trả lời đúng sự thật nhưng không có trong context vẫn bị trừ — đúng thứ ta muốn đo ở RAG. |

---

### 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** không phải lỗi thuật toán mà là **chuỗi lỗi môi trường im lặng**, mỗi lỗi đều "trông như" lỗi khác:
  1. `GROQ_MODEL=llama-3.3-70b-versatile` trả `model_not_found` — model đã bị Groq gỡ, dễ tưởng là sai API key.
  2. Judge trả **401 "Missing Authentication header"** dù key OpenRouter hợp lệ. Nguyên nhân thật: máy có sẵn biến môi trường hệ thống `OPENAI_API_KEY` của NVIDIA NIM, mà `load_dotenv()` **mặc định không ghi đè** biến đã tồn tại → `.env` của repo bị vô hiệu hoá âm thầm.
  3. Sửa xong 401 thì ra **402**: tài khoản OpenRouter `total_credits = 0`, model trả phí bị từ chối theo `max_tokens` yêu cầu.
  4. Neo4j: instance AuraDB trong `.env` đã bị xoá (DNS không phân giải). Chuyển sang Docker thì container chạy nhưng host **không kết nối được cổng 7687** — WSL2 đang ở `networkingMode=mirrored` làm port-forwarding của Docker Desktop hỏng; tệ hơn, Docker vẫn *bind* cổng 7687 nên khi chạy Neo4j native lại báo `Address already in use`.
- **Cách xử lý:** nguyên tắc chung là **đo, không đoán** — mỗi giả thuyết được kiểm chứng bằng một lệnh nhỏ nhất có thể trước khi sửa: gọi `/v1/models` để biết model nào còn sống, in `len(key)`/prefix để phát hiện key bị lấn át, `Test-NetConnection` để tách "Neo4j chưa sẵn sàng" khỏi "cổng không tới được", đọc `x-ratelimit-*` header để biết trần thật thay vì phỏng đoán.
- **Bài học lớn nhất về hiệu năng:** bản throttle đầu tiên của tôi ngủ trọn cửa sổ 60 giây mỗi khi chạm hạn mức. Đo lại qua header thì pipeline chỉ dùng ~2.400 token/phút trên trần 8.000 — **tự bóp mình còn 1/3 tốc độ**. Sau khi sửa thành "ngủ đúng lượng cần để giải phóng đủ token", throughput tăng ~2–3 lần. Thêm bài học thứ hai: bước LLM dài **phải checkpoint theo từng batch**; bản đầu chỉ ghi cache khi xong toàn bộ và đã làm mất ~25 phút khi phải dừng giữa chừng.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Đặc thù bài toán & lý do chọn giải pháp:** GraphRAG chỉ đáng giá khi câu hỏi **bắc cầu qua nhiều tài liệu** hoặc cần **trạng thái thay đổi theo thời gian**. Nếu phần lớn câu hỏi của đồ án là factoid nằm gọn trong một đoạn văn, Flat RAG rẻ hơn và thường không kém — kết quả nhóm `factoid` trong chính lab này là bằng chứng. Chi phí thật của GraphRAG nằm ở **indexing một lần** (LLM extraction), nên nó chỉ hoàn vốn khi corpus tương đối ổn định và số truy vấn đủ lớn.
- **Cấu trúc Node & Relation dự kiến:** giữ đúng kỷ luật đã dùng ở lab — tập label đóng, tập relation đóng, và **mọi cạnh bắt buộc mang provenance** (`source_chunk_id`, `published_date`, `evidence`). Không có provenance thì không thể debug được một câu trả lời sai đến từ đâu.
- **Chiến lược Entity Resolution:** ba tầng như lab (alias map → vector ANN → lexical guard), nhưng bổ sung guard theo miền của đồ án. Nguyên tắc rút ra: **vector similarity dùng để tạo candidate, không dùng để ra quyết định**; quyết định phải do luật có thể giải thích được đưa ra, và mọi quyết định phải ghi audit kèm lý do.
- **Chiến lược Super-node:** cắt tỉa theo thời gian như lab là mặc định hợp lý, nhưng phải nhớ rủi ro đối xứng: câu hỏi về quá khứ sẽ bị cắt mất bằng chứng. Hướng tốt hơn cho production là **cắt tỉa theo truy vấn** — ưu tiên cạnh có relation type liên quan tới câu hỏi, rồi mới tới cạnh mới nhất.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|---|---|---|
| Mức độ hiểu bài giảng GraphRAG | 4 | Nắm được vì sao hybrid (graph + vector) an toàn hơn graph-only: khi seed matching trượt, vector vẫn giữ hệ thống không rơi về rỗng. |
| Khả năng kiểm soát AI Coding Agent | 4 | Có 5 đề xuất bị từ chối kèm lý do kỹ thuật kiểm chứng được bằng số, không phải cảm tính. |
| Chất lượng đồ thị tri thức xây dựng | 3 | Schema và provenance đạt 100%, nhưng **độ phủ chỉ ~12% corpus** do trần token; đây là giới hạn ngân sách, không phải giới hạn kiến trúc. |
| Khả năng phân tích và debug hệ thống | 4 | Bốn lỗi môi trường nghiêm trọng được tách bạch và xử lý bằng đo đạc; hiệu năng pipeline được cải thiện dựa trên số đo rate-limit thật. |

---

## 📎 PHỤ LỤC A — BONUS Challenge A: Near-Dedup bằng MinHash + LSH

**Thuật toán:** shingle 5-gram ký tự → MinHash 128 hàm băm (`a·x + b mod (2³¹−1)`, chọn p đủ nhỏ để `a·x+b` không tràn `uint64` khi vector hoá) → LSH banding 16 band × 8 row → chỉ verify Jaccard thật trên các cặp cùng bucket.

**Hiệu quả loại bỏ so sánh thừa (đo thật):**

| Chỉ số | Giá trị |
|---|---|
| Số cặp nếu pairwise O(N²) | 2.231.328 |
| Candidate pair do LSH sinh ra | **304** |
| Tỉ lệ cắt giảm | **99,99%** |
| Cluster near-dup (>1 thành viên) | 102 bài trong 34 cluster |
| Phân loại | 98 `NEAR_DUP_KEEP` · 11 `REDUNDANT` (Jaccard ≥ 0,92) · 195 `REJECT_BELOW_THRESHOLD` |

**Quyết định kiến trúc: FLAG chứ không DROP** — và đây là chỗ có bằng chứng số thuyết phục nhất của cả bài. Row 33 và row 1746 (hai bản tin Aeris/Ericsson) có **Jaccard trên tiêu đề = 1,000** nhưng trên full text chỉ **0,419**. Nghĩa là:

- dedup theo **tiêu đề** — lối tắt tự nhiên nhất mà ai cũng nghĩ tới — sẽ **xoá mất một nửa cặp bằng chứng** của câu `G5000-02` (so sánh *"to acquire"* với *"has acquired"*);
- dedup theo full text ở ngưỡng 0,70 thì hai bài này **thậm chí không thành candidate**, nên corpus giữ nguyên được tín hiệu temporal.

**Định lượng tác động lên retrieval (trung thực):** trên 12 câu đánh giá, bật/tắt bộ lọc near-dup trong context Flat RAG cho kết quả **y hệt nhau** (5,92 "câu chuyện" riêng biệt và 1.674 ký tự context ở cả hai chế độ). Lý do: các bản gần trùng hiếm khi cùng lọt vào top-6 của một truy vấn cụ thể. Kết luận đúng phải là: **near-dedup ở lab này có giá trị như một cơ chế bảo vệ dữ liệu và một công cụ audit, chứ chưa chứng minh được lợi ích đo lường được ở tầng retrieval trên mẫu 12 câu.**

---

## 📎 PHỤ LỤC B — Những điểm CHƯA đạt và lý do

Liệt kê thẳng để người chấm không phải đi tìm:

| Tiêu chí | Trạng thái | Lý do |
|---|---|---|
| Audit Entity Resolution ≥ 10 dòng | **CHƯA ĐẠT — 5 dòng** | Đồ thị chỉ có 145 mention nên số cặp vượt ngưỡng audit 0,80 rất ít. Đây là hệ quả trực tiếp của ngân sách extraction 260 chunk, không phải do thiếu cơ chế: bảng audit có đủ 3 loại quyết định và 5 luật guard được kiểm chứng bằng unit test 5/5 pass trong notebook. |
| Super-node degree > 100 | **Không tồn tại trong dữ liệu** (max = 8) | Đồ thị quá nhỏ. Cell 5.1 kiểm chứng *cơ chế* bằng cách hạ ngưỡng xuống p95 thật và assert cả giới hạn 50 cạnh lẫn thứ tự "mới nhất trước". |
| Đánh giá toàn bộ 50 câu Golden | Chạy **12/50** câu | Trần token theo ngày của các nhà cung cấp miễn phí. Mẫu phân tầng giữ đủ 3 nhóm, mỗi nhóm 4 câu. |
| Bonus B (Community) & C (Self-correction) | **Không làm** | Ưu tiên hoàn thiện phần bắt buộc trong ngân sách token còn lại; code khung vẫn nằm trong notebook. |

**Ngân sách LLM thực tế đã dùng cho lần chạy cuối:** 68 lượt gọi · ~29.000 token Groq (extraction + generation) · ~31.200 token judge.

**Cấu hình 3 nhà cung cấp** (bắt buộc phải tách vì hạn mức token/ngày của Groq tính **theo từng model**, không phải theo tài khoản — dồn extraction và generation vào cùng một model đã làm cạn sạch 200k token và chặn đứng pipeline một lần):

| Vai trò | Model | Nhà cung cấp |
|---|---|---|
| Coreference + NER/RE | `openai/gpt-oss-120b` → `openai/gpt-oss-20b` | Groq |
| Generator (trả lời) | `openai/gpt-oss-20b` | Groq |
| LLM-as-a-Judge | `google/gemma-4-31b-it` | NVIDIA NIM |
