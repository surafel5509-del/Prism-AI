import "./globals.css";

export const metadata = {
  title: "Prism AI — Chat, Agent & Code Studio",
  description: "Think through ideas, plan with an AI agent, and create large code artifacts in a dedicated workspace.",
  applicationName: "Prism AI",
};

export default function RootLayout({ children }) {
  return <html lang="en"><body>{children}</body></html>;
}
