use crate::session::SESSION_ID_HEADER;
use crate::ExpectedSessionId;
use rmcp::model::{
    Annotations, CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, ErrorCode,
    Implementation, InitializeResult, ListToolsResult, PaginatedRequestParams, ProtocolVersion,
    Role, ServerCapabilities, ServerConfig, TextContent,
};
use rmcp::service::RequestContext;
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, RoleServer, ServerHandler};
use std::sync::Arc;
use tokio::task::JoinHandle;

pub const FAKE_CODE: &str = "test-uuid-12345-67890";

pub const TEST_IMAGE_B64: &str = include_str!("test_assets/test_image.b64").trim_ascii_end();

struct ValidatingService<S> {
    inner: S,
    expected_session_id: Arc<dyn ExpectedSessionId>,
}

impl<S> ValidatingService<S> {
    fn new(inner: S, expected_session_id: Arc<dyn ExpectedSessionId>) -> Self {
        Self {
            inner,
            expected_session_id,
        }
    }

    fn validate(&self, context: &RequestContext<RoleServer>) -> Result<(), McpError> {
        let actual = context.meta.get(SESSION_ID_HEADER).and_then(|v| v.as_str());
        self.expected_session_id
            .validate(actual)
            .map_err(|e| McpError::new(ErrorCode::INVALID_REQUEST, e, None))
    }
}

impl<S: ServerHandler> ServerHandler for ValidatingService<S> {
    fn get_info(&self) -> ServerConfig {
        self.inner.get_info()
    }

    async fn list_tools(
        &self,
        request: Option<PaginatedRequestParams>,
        context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, McpError> {
        self.validate(&context)?;
        self.inner.list_tools(request, context).await
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, McpError> {
        self.validate(&context)?;
        self.inner.call_tool(request, context).await
    }
}

#[derive(Clone, Default)]
pub struct McpFixtureServer;

#[tool_router]
impl McpFixtureServer {
    pub fn new() -> Self {
        Self
    }

    #[tool(description = "Get the code", annotations(read_only_hint = true))]
    fn get_code(&self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![ContentBlock::text(FAKE_CODE)]))
    }

    #[tool(description = "Get an image")]
    fn get_image(&self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![ContentBlock::image(
            TEST_IMAGE_B64,
            "image/png",
        )]))
    }

    #[tool(
        description = "Get audience-scoped content",
        annotations(read_only_hint = true)
    )]
    fn get_audience_content(&self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![
            ContentBlock::text("visible"),
            ContentBlock::Text(
                TextContent::new("provider-only")
                    .with_annotations(Annotations::default().with_audience(vec![Role::Assistant])),
            ),
        ]))
    }
}

#[tool_handler]
impl ServerHandler for McpFixtureServer {
    fn get_info(&self) -> ServerConfig {
        InitializeResult::new(ServerCapabilities::builder().enable_tools().build())
            .with_protocol_version(ProtocolVersion::V_2025_03_26)
            .with_server_info(Implementation::new("mcp-fixture", "1.0.0"))
            .with_instructions("Test server with code, image, and audience-scoped content tools.")
    }
}

pub struct McpFixture {
    pub url: String,
    handle: JoinHandle<()>,
}

impl Drop for McpFixture {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

impl McpFixture {
    pub async fn new(expected_session_id: Arc<dyn ExpectedSessionId>) -> Self {
        let service = StreamableHttpService::new(
            move || {
                Ok(ValidatingService::new(
                    McpFixtureServer::new(),
                    expected_session_id.clone(),
                ))
            },
            LocalSessionManager::default().into(),
            StreamableHttpServerConfig::default(),
        );
        let router = axum::Router::new().nest_service("/mcp", service);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let url = format!("http://{addr}/mcp");

        let handle = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });

        Self { url, handle }
    }
}
