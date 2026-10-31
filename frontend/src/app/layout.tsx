import type { Metadata } from "next";
import "./globals.css";
import "./genui.css";

export const metadata: Metadata = {
  title: "Fieldnote — Research workspace",
  description: "Search a local technical knowledge base and inspect the verified sources behind each answer.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
